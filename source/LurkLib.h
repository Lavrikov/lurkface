#ifndef _LurkLIB_H_
#define _LurkLIB_H_

#ifdef _MSC_VER
    #ifdef LURK_EXPORTS
    #define LURK_API extern "C" __declspec(dllexport) //This definition allows to preserve the same functions names as in .h files (VC can change the name during the compilation process)
    #else
    #define LURK_API extern "C" __declspec(dllimport)
    #endif
#else
    #ifdef LURK_EXPORTS
    #ifdef __GNUC__
    #define LURK_API extern "C" __attribute__((visibility("default")))
    #else // __GNUC__
    #define LURK_API extern "C"
    #endif // __GNUC__
    #else // CW_EXPORTS
    #define LURK_API extern "C"
    #endif // CW_EXPORTS 
#endif // _MCS_VER


// Defenition of external functions, you can use this code as dll in your project, with LurkLib.h
// please follow example code in main function.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function creates detection engine with name of caffe prototxt and caffemodel 
// images should be color RGB=16(8UC3)
// pathToDeployFile - path to caffe deploy file with the model layers definition
// pathToCaffeModelFile - path to caffemodel file with weights of model
// trsh_sort - confidence level to approve bbox as face for the tracker, you should adjust this parameter, to find balance between missing faces and false detections at your video
LURK_API void* lurkCreateDetector(const wchar_t * pathToDeployFile = L"lost_deploy.prototxt", const wchar_t * pathToCaffeModelFile = L"lost.caffemodel",
	const float trsh_sort = 0.2);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This function calls "ProcessFrame" method from the previously created detection engine
// - you should call it for every frame you want to process
// sdObject - engine
// width, height, stride - frames parameters
// frameDataRGB - frame in RBG style
// results - array with detected bboxes
LURK_API void lurkProcessFrame(void* cwObject, const char* uuid, const int* widths, const int* heights, const int* strides, const unsigned char* frameDataRGB, int* nResults, float** const results);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This function delete detection engine
LURK_API void lurkDestroy(void* cwObject);

#endif //_LurkLIB_H_
