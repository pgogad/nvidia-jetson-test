// create_engine.cpp
//
// This program creates TensorRT engine for the GoogLeNet model.
//
// Inputs:
//   deploy.prototxt
//   deploy.caffemodel
//
// Outputs:
//   deploy.engine

#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;


class IHostMemoryFromFile : public IHostMemory
{
    public:
        IHostMemoryFromFile(std::string filename);
        void* data() const { return mem; }
        std::size_t size() const { return s; }
        DataType type () const { return DataType::kFLOAT; } // not used
        void destroy() { free(mem); }
    private:
        void *mem{nullptr};
        std::size_t s;
};

IHostMemoryFromFile::IHostMemoryFromFile(std::string filename)
{
    std::ifstream infile(filename, std::ifstream::binary | std::ifstream::ate);
    s = infile.tellg();
    infile.seekg(0, std::ios::beg);
    mem = malloc(s);
    infile.read(reinterpret_cast<char*>(mem), s);
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"./"};
    return locateFile(input, dirs);
}

void caffeToTRTModel(const std::string& deployFile,             // name for caffe prototxt
                     const std::string& modelFile,              // name for model
                     const std::vector<std::string>& outputs,   // network outputs
                     unsigned int maxBatchSize,                 // batch size - NB must be at least as large as the batch we want to run with)
                     IHostMemory *&trtModelStream)
{
    // create API root class - must span the lifetime of the engine usage
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    // parse the caffe model to populate the network, then set the outputs
    ICaffeParser* parser = createCaffeParser();

    bool useFp16 = builder->platformHasFastFp16();

    // create a 16-bit model if it's natively supported
    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;
    const IBlobNameToTensor *blobNameToTensor =
        parser->parse(locateFile(deployFile).c_str(),  // caffe deploy file
                      locateFile(modelFile).c_str(),   // caffe model file
                      *network,                        // network definition that the parser will populate
                      modelDataType);

    assert(blobNameToTensor != nullptr);
    // the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(64 << 20);

    // set up the network for paired-fp16 format if available
    if (useFp16) {
#if NV_TENSORRT_MAJOR == 3
        builder->setHalf2Mode(true);
#else   // NV_TENSORRT_MAJOR >= 4
        builder->setFp16Mode(true);
#endif  // NV_TENSORRT_MAJOR
    }

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    parser->destroy();
    network->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
}

void giestream_to_file(IHostMemory *trtModelStream, const std::string filename)
{
    assert(trtModelStream != nullptr);
    std::ofstream outfile(filename, std::ofstream::binary);
    assert(!outfile.fail());
	outfile.write(reinterpret_cast<char*>(trtModelStream->data()), trtModelStream->size());
    outfile.close();
}

void file_to_giestream(const std::string filename, IHostMemory *&trtModelStream)
{
    trtModelStream = new IHostMemoryFromFile(filename);
}

void verify_engine(std::string det_name)
{
    std::stringstream ss;
    ss << det_name << ".engine";
    IHostMemory *trtModelStream{nullptr};
    file_to_giestream(ss.str(), trtModelStream);

    // create an engine
    IRuntime* infer = createInferRuntime(gLogger);
    assert(infer != nullptr);
    ICudaEngine* engine = infer->deserializeCudaEngine(
        trtModelStream->data(),
        trtModelStream->size(),
        nullptr);
    assert(engine != nullptr);

    assert(engine->getNbBindings() == 2);
    std::cout << "Bindings for " << det_name << " after deserializing:"
              << std::endl;
    for (int bi = 0; bi < 2; bi++) {
#if NV_TENSORRT_MAJOR == 3
        DimsCHW dim = static_cast<DimsCHW&&>(engine->getBindingDimensions(bi));
        if (engine->bindingIsInput(bi) == true) {
            std::cout << "  Input  ";
        } else {
            std::cout << "  Output ";
        }
        std::cout << bi << ": " << engine->getBindingName(bi) << ", "
                  << dim.c() << "x" << dim.h() << "x" << dim.w()
                  << std::endl;
#else   // NV_TENSORRT_MAJOR >= 4
        Dims3 dim = static_cast<Dims3&&>(engine->getBindingDimensions(bi));
        if (engine->bindingIsInput(bi) == true) {
            std::cout << "  Input  ";
        } else {
            std::cout << "  Output ";
        }
        std::cout << bi << ": " << engine->getBindingName(bi) << ", "
                  << dim.d[0] << "x" << dim.d[1] << "x" << dim.d[2]
                  << std::endl;
#endif  // NV_TENSORRT_MAJOR
    }
    engine->destroy();
    infer->destroy();
    trtModelStream->destroy();
}

int main(int argc, char** argv)
{
    IHostMemory *trtModelStream{nullptr};

    std::cout << "Building deploy.engine, maxBatchSize = 1" << std::endl;
    caffeToTRTModel("deploy.prototxt",
                    "deploy.caffemodel",
                    std::vector <std::string> { "prob" },
                    1,  // batch size
                    trtModelStream);
    giestream_to_file(trtModelStream, "deploy.engine");
    trtModelStream->destroy();
    //delete trtModelStream;

    shutdownProtobufLibrary();

    std::cout << std::endl << "Verifying engine..." << std::endl;
    verify_engine("deploy");
    std::cout << "Done." << std::endl;
    return 0;
}
