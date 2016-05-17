#ifndef _LIBSVM_H
#define _LIBSVM_H

#define LIBSVM_VERSION 320

#ifdef __cplusplus
extern "C" {  // 表示C在C++编译环境中进行编译指定
#endif

extern int libsvm_version;

struct svm_node  // 用来存储单一向量中的单个特征
{
	int index;  // 特征下标
	double value;  // 特征值
};

struct svm_problem  // 用来存储本次参加运算的所有样本（数据集）及其所属类别
{
	int l;  // 样本总数
	double *y;  // 指向样本所属类别数组
	struct svm_node **x;  // 指向一个存储内容为指针的数组，故可以使用x[i][j]访问其中的某一个元素
};

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */  // SVM类型，分类问题：C_SVC,NU_SVC；回归问题：EPSILON_SVR,NU_SVR；ONE_CLASS：分布估计
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */  // SVM核的类型，各种核类型式子
/*
* 相关核函数解释
* K( x(i), x(j) ) = x(i).T * x(j)
* K( x(i), x(j) ) = ( gamma * x(i).T * x(j) + coef0 ) ^ degree, gamma > 0
* K( x(i), x(j) ) = exp( -gamma * |x(i) - x(j)| ^ 2 ), gamma > 0
* K( x(i), x(j) ) = tanh( gamma * x(i).T * x(j) + coef0 )
*/

struct svm_parameter  // SVM相关参数
{
	int svm_type;  // SVM类型
	int kernel_type;  // SVM核函数类型
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */  // 指定训练所需要的内存，有个默认值
	double eps;	/* stopping criteria */  // 暂定条件，及其暂停阈值（需要查看文献，了解其意义）
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */  // 惩罚因子
	int nr_weight;		/* for C_SVC */  // 权重的数目，目前在实例代码中只有两个值，一个是默认0，另一个是svm_binary_svc_probability函数中使用数值2
	int *weight_label;	/* for C_SVC */  // 权重label
	double* weight;		/* for C_SVC */  // 权值值（估计是：weight_label--->weight 两两相互对应）
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */  // 启发式缩减，指明训练过程中是否使用压缩
	int probability; /* do probability estimates */  // 概率估计，指明是否要做概率估计
};

//
// svm_model
// 
struct svm_model  // svm_model用于保存训练后的训练模型，当然原来的训练参数也必须保留
{
	struct svm_parameter param;	/* parameter */  // SVM参数
	int nr_class;		/* number of classes, = 2 in regression/one class svm */  // 类别数
	int l;			/* total #SV */  // 支持向量数
	struct svm_node **SV;		/* SVs (SV[l]) */  // 保存（支持向量）的指针，至于支持向量的内容，如果是从文件中读取，内容会
                                                   // 额外保留；如果是直接训练得来，则保留在原来的训练集中。如果训练完成后需要预报，原来的
                                                   // 训练集内存不可以释放。
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */  // 相当于判别函数中的alpha
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */  // 相当于判别函数中的b
	double *probA;		/* pariwise probability information */  // 概率信息
	double *probB;
	int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */  // 标识支持向量

	/* for classification only */

	int *label;		/* label of each class (label[k]) */  // 每个类别中的label
	int *nSV;		/* number of SVs for each class (nSV[k]) */  // 每个类别中的支持向量数
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */  // 竟然是为1，难道是概率表示么？
	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/  // 1则表示是导入Model，0则表示训练的Model
				/* 0 if svm_model is created by svm_train */
};

// 最主要的驱动函数，训练数据，输入训练集及其模型参数
struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
// SVM做交叉验证，输入训练集，模型参数，nr_fold（n-cv数），target目标值
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);
// 保存训练好的模型到文件，输入文件名，训练好的模型
int svm_save_model(const char *model_file_name, const struct svm_model *model);
// 从文件中把训练好的模型读取到内存中，输入模型文件名
struct svm_model *svm_load_model(const char *model_file_name);
// 获取SVM的类型
int svm_get_svm_type(const struct svm_model *model);
// 得到数据集的类别数（必须经过训练得到模型后才可以用）
int svm_get_nr_class(const struct svm_model *model);
// 得到数据集的类别标号（必须经过训练得到模型后才可以用）
void svm_get_labels(const struct svm_model *model, int *label);
// 
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
//
int svm_get_nr_sv(const struct svm_model *model);
//
double svm_get_svr_probability(const struct svm_model *model);
// 用训练好的模型预报样本的值，输出结果保留到数组中（并非接口函数），输入模型，数据值，保存输出结果矩阵
double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
// 预报某一样本的值
double svm_predict(const struct svm_model *model, const struct svm_node *x);
//
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);
// 消除训练的模型，释放资源
void svm_free_model_content(struct svm_model *model_ptr);
//
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
// 
void svm_destroy_param(struct svm_parameter *param);
// 检测输入的参数，保证后面的训练能正常进行
const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
//
int svm_check_probability_model(const struct svm_model *model);
// 设置输出结果函数？
void svm_set_print_string_function(void (*print_func)(const char *));

#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
