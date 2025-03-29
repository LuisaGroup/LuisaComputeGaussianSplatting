#pragma once 
/**
 * @file config.h
 * @brief LCGS configuration header.
 * @author sailing-innocent
 * @date 2025-03-29
 */


#ifdef LCGS_DLL_EXPORTS
#define LCGS_API __declspec(dllexport)
#else
#define LCGS_API __declspec(dllimport)
#endif