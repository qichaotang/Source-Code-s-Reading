#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"

int print_null(const char *s,...) {return 0;}  // 标准输出,stdout，不输出结果

static int (*info)(const char *fmt,...) = &printf;

struct svm_node *x;  // 单一向量中的单个特征
int max_nr_attr = 64;

struct svm_model* model;  // 训练的模型
int predict_probability=0;

static char *line = NULL;
static int max_line_len;

// 读取文件中数据到line中
static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

// 预测数据函数，预测输入数据值input，将返回结果输出到output中
void predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);  // 获取模型svm类型
	int nr_class=svm_get_nr_class(model);  // 获取模型类别class数
	double *prob_estimates=NULL;  // 概率估计
	int j;

	if(predict_probability)  // 此时表示预测概率
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)  // SVM回归模型
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));  // 直接回归预测概率值
		else
		{
			int *labels=(int *) malloc(nr_class*sizeof(int));
			svm_get_labels(model,labels);  // 获取模型类别
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");		
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
			free(labels);
		}
	}

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)  // 读取待预测数据值
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		label = strtok(line," \t\n");  // 分解字符串为多个字符串，即字符串数组
		if(label == NULL) // empty line
			exit_input_error(total+1);  // 无label，则错误输出

		target_label = strtod(label,&endptr);  // 将label字符串转化为float,每次调用此个函数，都需要做相应的错误处理
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");  // 进行字符分割，提取训练数据集
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)  // 
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))  // 
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;

		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))  // SVM分类模型，进行概率预测
		{
			predict_label = svm_predict_probability(model,x,prob_estimates);  // 预测预测集x，对应的概率值
			fprintf(output,"%g",predict_label);
			for(j=0;j<nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
			predict_label = svm_predict(model,x);  // SVM分类模型，类别预测，进行分类
			fprintf(output,"%g\n",predict_label);
		}

		if(predict_label == target_label)  // 表示预测正确
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);  // 错误方差？
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)  // SVM回归模型
	{
		info("Mean squared error = %g (regression)\n",error/total);  // 均值平方误差
		info("Squared correlation coefficient = %g (regression)\n",  // 相关度
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		info("Accuracy = %g%% (%d/%d) (classification)\n",  // 准确率
			(double)correct/total*100,correct,total);
	if(predict_probability)
		free(prob_estimates);
}

// 预测帮助函数
void exit_with_help()
{
	printf(
	"Usage: svm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;  // 开头处-""，表示是否为参数类型标识，
		++i;
		switch(argv[i-1][1])  // 根据参数标识，转换参数值为正确类型或相应设置
		{
			case 'b':
				predict_probability = atoi(argv[i]);
				break;
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	if(i>=argc-2)
		exit_with_help();

	input = fopen(argv[i],"r");  // 读取输入文件中数据
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");  // 读取输出文件中数据
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model=svm_load_model(argv[i+1]))==0)  // 从模型文件中加载模型
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	if(predict_probability)  // 表示是预测概率，还是类别（值）
	{
		if(svm_check_probability_model(model)==0)  // 检测概率预测模型
		{
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else
	{
		if(svm_check_probability_model(model)!=0)
			info("Model supports probability estimates, but disabled in prediction.\n");
	}

	predict(input,output);  // 进行预测
	svm_free_and_destroy_model(&model);  // 释放并且销毁模型
	free(x);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
}
