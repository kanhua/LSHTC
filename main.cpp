#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svm.h"
#include "omp.h"
#include <ctype.h>

#define NUM_THREADS 2

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;

static char *line = NULL;
static int max_line_len;


void read_problem(const char *filename);
double spssum(struct svm_problem &prob,int row);
void fillrowptr(struct svm_problem &prob1);
double* spsmulti(struct svm_problem &prob1,struct svm_problem &prob2,double *resultmtx);

int main(void)
{
	const char *testfile="train-sk-min.txt";
	
	//read_problem("a1adata.dat");
	read_problem(testfile);
	//read_problem("simplemtx.txt");
	printf("file read!\n");
	

	fillrowptr(prob);
	printf("%f\n",omp_get_wtime());
	double *result=new double[1];
	result=spsmulti(prob,prob,result);
	printf("%f\n",omp_get_wtime());

	//print the result
	FILE *ofp = fopen("output.dat", "w");

	for (int i=0;i<prob.l;i++)
	{
		for(int j=0;j<prob.l;j++)
		{
			fprintf(ofp,"%3.0f",result[i*prob.l+j]);
			if (j<prob.l-1)
				fprintf(ofp,",");
			
		}
		fprintf(ofp,"\n");
	}

	int d=0;
	scanf("?",&d);
}

double spssum(struct svm_problem &prob1,int row)
{
	//Calculate the sum of a particular row

	double sum=0;
	for (int i=0;i<prob1.rowptr[row]-1;i++)
	{
		sum=sum+prob1.x[row][i].value;
	}

	return sum;
}

double *spsmulti(struct svm_problem &prob1,struct svm_problem &prob2,double *resultmtx)
{
	//This function calcualtes the matrix: prob1 *transpose(prob2)
	omp_set_num_threads(NUM_THREADS);
	// determine the dimension of the matrix
	
	// Allocate the space for matrix
	resultmtx=new double[prob1.l*prob2.l];
	// check the dimension(optional for now)
	//matrix multiplication

	#pragma omp parallel for
	for(int i=0;i<prob1.l;i++)
	{
		for (int j=0;j<prob2.l;j++)
		{
			//calcualte the inner product of prob1[i,:] and prob2[j,:]
			//run through the element in each row
			
			double sum=0;
			for (int k=0;k<prob1.rowptr[i]-1;k++)
			{
				for (int m=0;m<prob2.rowptr[j]-1;m++)
				{
					if(prob1.x[i][k].index==prob2.x[j][m].index)
					{
						sum=sum+(prob1.x[i][k].value*prob2.x[j][m].value);
					}	
				}
			}
		
			resultmtx[i*prob1.l+j]=sum;
			
		}
	}

	return resultmtx;
}

void fillrowptr(struct svm_problem &prob1)
{
	prob1.rowptr=Malloc(int,prob.l);
	for(int i=0;i<prob.l;i++)
	{
		int index=-1;
		int cols=0;
		do
		{
			index=prob.x[i][cols].index;
			cols++;
		}while(index!=-1);
		
		prob1.rowptr[i]=cols;
		
	}
}

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

void read_problem(const char *filename)
{

	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;
	int errno;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	
	//Determine the number of elements the lables
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		
		
		readline(fp); // read the next line in the file into global "line" variable
		
		prob.x[i] = &x_space[j];
		
		//below is the procedure that extracts the mulilabel(add by KH)
		
		label = strtok(line,",");
		char *leftline=Malloc(char,max_line_len);
		leftline=strcpy(leftline,line);
		while(1)
		{
			label=strtok(NULL,",");
			if (label==NULL)
				break;
			leftline=strcpy(leftline,label);
		}

		//end procedure
		label = strtok(leftline," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}