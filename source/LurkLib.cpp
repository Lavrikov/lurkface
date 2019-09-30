#include "LurkLib.h"
#include "Detector.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LURK_API void* lurkCreateDetector(const wchar_t * pathToDeployFile, const wchar_t * pathToCaffeModelFile, const float trsh_sort)
{
	DetectorSSD* SSD = nullptr;
	SSD = new DetectorSSD(pathToDeployFile, pathToCaffeModelFile, trsh_sort, "123,104,117");// напрямую вызывать конструктор при создании обьекта типа SSDdetector* не возможно т.к. конструктор не принимает тип wchar*

	return SSD;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LURK_API void lurkProcessFrame(void* ssdObject, const char* uuid, const int* widths, const int* heights, const int* strides, const unsigned char* frameDataRGB, int* nResults, float** results)
{
	DetectorSSD* SSD = static_cast<DetectorSSD*>(ssdObject);//приводим в соответсвие тип ссылки void* к типу DetectorSSD*, тогда становятся видны методы этого класса, а инече компилятор не знает что за обьект на который мы ссылаемся
    return SSD->Detect(uuid, widths, heights, strides, frameDataRGB, nResults, results);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LURK_API void lurkDestroy(void* SSD)
{
	delete SSD;
}
