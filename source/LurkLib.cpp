#include "LurkLib.h"
#include "Detector.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LURK_API void* lurkCreateDetector(const wchar_t * pathToDeployFile, const wchar_t * pathToCaffeModelFile, const float trsh_sort)
{
	DetectorSSD* SSD = nullptr;
	SSD = new DetectorSSD(pathToDeployFile, pathToCaffeModelFile, trsh_sort, "123,104,117");

	return SSD;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LURK_API void lurkProcessFrame(void* ssdObject, const char* uuid, const int* widths, const int* heights, const int* strides, const unsigned char* frameDataRGB, int* nResults, float** results)
{
	DetectorSSD* SSD = static_cast<DetectorSSD*>(ssdObject);//Converts between types (from void* to DetectorSSD*). So, we can call methods of engine.
    return SSD->Detect(uuid, widths, heights, strides, frameDataRGB, nResults, results);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LURK_API void lurkDestroy(void* SSD)
{
	delete SSD;
}
