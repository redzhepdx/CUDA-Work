#ifndef _UTILS_H
#define _UTILS_H

#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }
  return result;
}

inline void codeFlowControl(std::string s){
	std::cout << s << std::endl;
}

inline void splitImage2Channels(cv::Mat img, uchar *r_ch, uchar *g_ch, uchar *b_ch){
    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            r_ch[row * img.cols + col] = img.at<cv::Vec3b>(row, col)[2];
            g_ch[row * img.cols + col] = img.at<cv::Vec3b>(row, col)[1];
            b_ch[row * img.cols + col] = img.at<cv::Vec3b>(row, col)[0];
        }
    }
}

inline void vector2Image(uchar *res, cv::Mat &img, int rows, int cols){
    for(int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            img.at<uchar>(row,col) = res[row * cols + col];
        }
    }
}

inline void vector2ImageRGB(uchar *r, uchar *g, uchar *b, cv::Mat &img, int rows, int cols){
	for(int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            img.at<cv::Vec3b>(row, col)[2] = r[row * cols + col];
			img.at<cv::Vec3b>(row, col)[1] = g[row * cols + col];
			img.at<cv::Vec3b>(row, col)[0] = b[row * cols + col];
        }
    }
}

#endif
