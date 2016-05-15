/*
 *  Copyright 2009-2010 APEX Data & Knowledge Management Lab, Shanghai Jiao Tong University
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*!
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "apex_svd_data.h"  // 并不在此个目录下，不会出现BUG么？解答：此个文件好像是在主目录下调用，编译，故此文件直接可以引用主目录下头文件
#include "apex-utils/apex_utils.h"

using namespace apex_svd;  // apex_svd：表示数据格式的命名空间

int main( int argc, char *argv[] ){
    if( argc < 3 ){  // batch_size 表示：mini-batch size（分批梯度大小）
        printf("Usage:make_feature_buffer <input> <output> [options...]\n"\
               "options: -batch_size batch_size, -scale_score scale_score\n"\
               "example: make_feature_buffer input output -batch_size 100 -scale_score 1\n"\
               "\tmake a buffer used for svd-feature\n"\
               "\tbatch_size is the mini-batch size for the data entry, must be set smaller than total number of entrys\n"\
               "\tscale_score will divide the score by scale_score, we suggest to scale the score to 0-1 if it's too big\n");
        return 0; 
    }
    int batch_size = 1000;  // 初始化batch_size值，
    IDataIterator<SVDFeatureCSR::Elem> *loader = create_csr_iterator( input_type::TEXT_FEATURE );
    loader->set_param( "scale_score", "1.0" );  // 设置参数值，即初始化scale_score值

    time_t start = time( NULL );
    for( int i = 3; i < argc; i ++ ){  // 读取输入的参数
        if( !strcmp( argv[i], "-batch_size") ){
            batch_size = atoi( argv[++i] ); continue;  // 根据读入值，设置batch_size值
        }
        if( !strcmp( argv[i], "-scale_score") ){
            loader->set_param( "scale_score", argv[++i] ); continue;  // 根据读入值，设置scale_score值
        }
    }
    
    loader->set_param( "data_in", argv[1] );  // 设置输入数据文件名
    loader->init();  // loader进行初始化
    printf("start creating buffer...\n");
    create_binary_buffer( argv[2], loader, batch_size );  // 创建buffer文件，并且输出到argv[2]文件中去
    printf("all generation end, %lu sec used\n", (unsigned long)(time(NULL) - start) );
    delete loader;
    return 0;
}
