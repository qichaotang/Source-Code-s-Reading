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
#include <climits>

#include "apex_svd.h"
#include "apex-utils/apex_task.h"
#include "apex-utils/apex_utils.h"
#include "apex-utils/apex_config.h"
#include "apex-tensor/apex_random.h"

namespace apex_svd{
    class SVDTrainTask : public apex_utils::ITask{  // 集成抽象类apex_utils::ITask
    private:
        // type of model  // 模型类型
        SVDTypeParam mtype;  // mtype表示SVD类型参数，其默认值为basicMF.conf配置文件中参数值
        ISVDTrainer   *svd_trainer;  // 表示 ISVDTrainer 这个训练参数
    private:
        int input_type;  // 表示输入文件类型,官网上存在两种数据类型
        IDataIterator<SVDFeatureCSR::Elem> *itr_csr;  // 输入数据文件格式类迭代器
        IDataIterator<SVDPlusBlock>        *itr_plus;
    private:
        // initialize end
        int init_end;
        // name of job
        char name_job[ 256 ];
        // name of configure file
        char name_config[ 256 ];  // 配置文件名
        // 0 = new layer, 1 = continue last layer's training
        int task;
        // continue from model folder
        int continue_training;
        // whether to be silent 
        int silent;        
        // input model name 
        char name_model_in[ 256 ];  // 输入模型名
        // start counter of model
        int start_counter;                
        // folder name of output  
        char name_model_out_folder[ 256 ];  // 输出模型目录
    private:
        float print_ratio;
        int   num_round, train_repeat, max_round;
    private:
        apex_utils::ConfigSaver cfg;  // 存储配置文件，提取到svd_trainer中去
    private:
        inline void reset_default(){  // 设置参数默认值
            init_end = 0;
            this->svd_trainer = NULL;  // svd_trainer默认为NULL
            this->input_type  = input_type::BINARY_BUFFER;  // 默认为0，表示输入值为数据格式为二进制类型的文件
            this->itr_csr     = NULL;
            this->itr_plus    = NULL;
            strcpy( name_config, "config.conf" );
            strcpy( name_job, "" );
            strcpy( name_model_out_folder, "models" );
            print_ratio = 0.05f;  // print_ratio参数初始化为0.05
            train_repeat = 1;  // 模型重复次数
            num_round = 10;  //  默认初始化为10，模型迭代次数，输入时候，其值为40         
            task = silent = start_counter = 0;  // task参数默认为0,start_counter为0，
            max_round = INT_MAX;  // 最大的迭代次数
            continue_training = 0;  // 初始化continue_training值为0            
        }
    public:
        SVDTrainTask(){
            this->reset_default();
        }
        virtual ~SVDTrainTask(){           
            if( init_end ){
                delete svd_trainer;
                if( itr_csr != NULL ) delete itr_csr;
                if( itr_plus!= NULL ) delete itr_plus;
            }
        }
    private:
        inline void set_param_inner( const char *name, const char *val ){  // 根据参数名与参数值进行设定参数值
            if( !strcmp( name,"task"   ))             task    = atoi( val ); 
            if( !strcmp( name,"seed"   ))             apex_random::seed( atoi( val ) ); 
            if( !strcmp( name,"continue"))            continue_training = atoi( val ); 
            if( !strcmp( name,"max_round"))           max_round = atoi( val ); 
            if( !strcmp( name,"start_counter" ))      start_counter = atoi( val );
            if( !strcmp( name,"model_in" ))           strcpy( name_model_in, val ); 
            if( !strcmp( name,"model_out_folder" ))   strcpy( name_model_out_folder, val ); 
            if( !strcmp( name,"num_round"  ))         num_round    = atoi( val ); 
            if( !strcmp( name,"train_repeat"  ))      train_repeat = atoi( val );
            if( !strcmp( name, "silent") )            silent     = atoi( val );
            if( !strcmp( name, "job") )               strcpy( name_job, val ); 
            if( !strcmp( name, "print_ratio") )       print_ratio= (float)atof( val );            
            if( !strcmp( name, "input_type"  ))       input_type = atoi( val ); 
            mtype.set_param( name, val );  // 初始化 mtype 参数，即初始化配置文件中参数值
        }
        
        inline void configure( void ){  // 配置文件设置，此时name_config = "basicMF.conf" 此个参数值
            apex_utils::ConfigIterator itr( name_config );
            while( itr.next() ){
                cfg.push_back( itr.name(), itr.val() );  // 读取配置文件中的参数属性-参数值，并都入栈,与此同时初始化 cfg 这个配置类
            }

            cfg.before_first();
            while( cfg.next() ){
                set_param_inner( cfg.name(), cfg.val() );  // 对于配置文件中的参数，并且进行初始化 SVDTrainTask 中的参数值
            }
            // decide format type
            mtype.decide_format( input_type == 2 ? svd_type::USER_GROUP_FORMAT : svd_type::AUTO_DETECT );  // input_type默认值为0，则表示输入数据格式为后者svd_type::AUTO_DETECT
        }

        // configure iterator
        inline void configure_iterator( void ){  // 配置文件迭代器
            if( mtype.format_type == svd_type::USER_GROUP_FORMAT ){
                this->itr_plus = create_plus_iterator( input_type );
            }else{
                this->itr_csr = create_csr_iterator( input_type );  // 根据输入文件格式,进行初始化相应的格式的文件参数,input_type默认为0，表示二进制文件格式,并且返回相应文件格式的类
            }        

            cfg.before_first();
            while( cfg.next() ){
                if( itr_csr != NULL  ) itr_csr->set_param( cfg.name(), cfg.val() );  // 又设置相应参数值
                if( itr_plus != NULL ) itr_plus->set_param( cfg.name(), cfg.val() );
            }
                
            if( itr_csr != NULL ) itr_csr->init();  // 
            if( itr_plus!= NULL ) itr_plus->init();
        }

        inline void configure_trainer( void ){
            cfg.before_first();
            while( cfg.next() ){
                svd_trainer->set_param( cfg.name(), cfg.val() );  // 初始化svd_trainer训练时候的配置文件值
            }
        }
        
        // load in latest model from model_folder
        inline int sync_latest_model( void ){
            FILE *fi = NULL, *last = NULL;           
            char name[256];
            int s_counter = start_counter;  // 
            do{
                if( last != NULL ) fclose( last );
                last = fi;
                sprintf(name,"%s/%04d.model" , name_model_out_folder, s_counter++ );
                fi = fopen64( name, "rb");                
            }while( fi != NULL ); 

            if( last != NULL ){
                apex_utils::assert_true( fread( &mtype, sizeof(SVDTypeParam), 1, last ) > 0, "loading model" );
                svd_trainer = create_svd_trainer( mtype );  // 创建SVD模型训练
                svd_trainer->load_model( last );
                start_counter = s_counter - 1;
                fclose( last );
                return 1;
            }else{
                return 0;
            }
        }
        inline void load_model( void ){
            FILE *fi = apex_utils::fopen_check( name_model_in, "rb" );
            // load model from file 
            apex_utils::assert_true( fread( &mtype, sizeof(SVDTypeParam), 1, fi ) > 0, "loading model" );
            svd_trainer = create_svd_trainer( mtype );
            svd_trainer->load_model( fi );
            fclose( fi );
        }
        
        inline void save_model( void ){
            char name[256];
            sprintf(name,"%s/%04d.model" , name_model_out_folder, start_counter ++ );  // 模型路径名,即模型名,start_counter之后加1
            FILE *fo  = apex_utils::fopen_check( name, "wb" );  // 读取模型名文件            
            fwrite( &mtype, sizeof(SVDTypeParam), 1, fo );  // 将mtype写入fo中
            svd_trainer->save_model( fo );  // 存储模型文件---> 跳转到svd_trainer所指向的类函数
            fclose( fo );
        }
        
        
        inline void init( void ){  // 执行run_task时候进行初始化配置文件参数          
            // configure the parameters
            this->configure();  // 初始化配置文件参数，此时name_config = "basicMF.conf" 此个参数值           
            // configure trainer
            if( continue_training != 0 && sync_latest_model() != 0 ){  // continue_training默认初始值为0
                this->configure_trainer();
            }else{                                   
                continue_training = 0; 
                switch( task ){  // task参数默认为0，则表示需要训练模型
                case 0:  // 训练模型
                    svd_trainer = create_svd_trainer( mtype );  // mtype默认初始化值为 basicMF.conf 配置文件中参数值，默认创建模型类型为:SVDFeature
                    // 从此刻开始，svd_trainer所指向类别为--->SVDFeature类型，在Apex_svd_base.h文件中
                    // 获取svd_trainer的类名看看typeid(A).name()
                    this->configure_trainer();  // svd_trainer中参数初始化，初始化相应的函数参数值
                    svd_trainer->init_model();  // svd_trainer中初始化模型
                    break;
                case 1:  // 加载模型
                    this->load_model(); 
                    this->configure_trainer();
                    break;
                default: apex_utils::error("unknown task");
                }
            }
            this->configure_iterator();
            svd_trainer->init_trainer();  // 初始化训练器
            this->init_end = 1;  //            
        }     

        template<typename DataType>
        inline void update( int r, unsigned long elapsed, time_t start, IDataIterator<DataType> *itr ){  // *itr此个参数表示数据类型
            size_t total_num = itr->get_data_size() * train_repeat;  //

            // exceptional case when iterator didn't provide data count
            if( total_num == 0 ) total_num = 1; 

            size_t print_step = static_cast<size_t>( floorf(total_num * print_ratio ));  // floorf返回相应下限值
            if( print_step <= 0 ) print_step = 1;
            size_t sample_counter = 0;
            DataType dt;
            for( int j = 0; j < train_repeat; j ++ ){  // 重复训练次数
                while( itr->next( dt ) ){  // itr，迭代器，遍历整个数据集
                    // Debug--->开始根据数据进行更新模型?看看具体原理!!!
                    svd_trainer->update( dt );  // 根据输入数据进行更新模型?此时数据格式为:Elem类型, SVDFeature::update()
                    if( sample_counter  % print_step == 0 ){
                        if( !silent ){
                            elapsed = (unsigned long)(time(NULL) - start); 
                            printf("\r                                                                     \r");
                            printf("round %8d:[%05.1lf%%] %lu sec elapsed", 
                               r , (double)sample_counter / total_num * 100.0, elapsed );
                            fflush( stdout );
                        }
                    }
                    sample_counter ++;
                }  // 遍历完所有数据集
                svd_trainer->finish_round();  // 什么操作也没有!!!
                itr->before_first();                    
            }
        }

    public:
        virtual void set_param( const char *name , const char *val ){
            cfg.push_back_high( name, val );
        }
        virtual void set_task ( const char *task ){
            strcpy( name_config, task );  // name_config = "basicMF.conf" 此个参数值
        }
        virtual void print_task_help( FILE *fo ) const {
            printf("Usage:<config> [xxx=xx]\n");
        }
        virtual void run_task( void ){  // 执行训练任务
            this->init();  // 初始化参数，name_config = "basicMF.conf" 此个参数值，整个配置文件，有点长的样子svd_trainer类型发生了变化，变为
            if( !silent ){  // silent默认值为0
                printf("initializing end, start updating\n");
            }
            time_t start    = time( NULL );
            unsigned long elapsed = 0;
            
            if( continue_training == 0 ){  // continue_training默认初始值为0
                this->save_model();  // 开启训练模型，并且进行模型存储
            }
            
            int cc = max_round; 
            while( start_counter <= num_round && cc -- ) {  // 模型迭代训练,start_counter为1，在进入此个循环时候
                // Debug start_counter,此个值是否发生改变了!!!确实是发生了改变
                svd_trainer->set_round( start_counter -1 );  // 在basic_MF的训练过程中，此值作用不大，无法执行

                if( itr_csr != NULL )  // 此时为itr_csr类型
                    this->update( start_counter-1, elapsed, start, itr_csr );  // 更新模型操作
                if( itr_plus != NULL )
                    this->update( start_counter-1, elapsed, start, itr_plus );

                elapsed = (unsigned long)(time(NULL) - start);  // 模型消耗时间
                this->save_model();  // 模型存储，此个模型训练过程到底是怎么样的?            }

            if( !silent ){
                printf("\nupdating end, %lu sec in all\n", elapsed );
            }                        
        }        
    };
};

int main( int argc, char *argv[] ){
    apex_random::seed( 10 );
    apex_svd::SVDTrainTask tsk;  // SVD模型训练,默认初始化相关参数
    return apex_utils::run_task( argc, argv, &tsk );  // 进行模型训练，此处tsk类型为apex_svd::SVDTrainTask，此个类继承自apex_svd::ITask这个抽象类,总定位到此个抽象类的函数接口
}

