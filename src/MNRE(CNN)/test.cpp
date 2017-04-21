#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <map>
#include <cmath>
#include <pthread.h>
#include <iostream>

#include<assert.h>
#include<ctime>
#include<sys/time.h>

#include "init.h"
#include "test.h"
#include "test_rel.h"

void read_para()
{
	cout<<20<<endl;
	FILE *fin = fopen(("./out/matrixW1+B1.txt"+version).c_str(), "r");
	fscanf(fin,"%d%d%d%d", &dimensionC, &dimension, &window, &dimensionWPE);
	for (int idx = 0; idx<num_language; idx++)
	for (int i = 0; i < dimensionC; i++) {
		for (int j = 0; j < dimension * window; j++)
			fscanf(fin, "%f",&matrixW1[idx][i* dimension*window+j]);
		for (int j = 0; j < dimensionWPE * window; j++)
			fscanf(fin, "%f",&matrixW1PositionE1[idx][i* dimensionWPE*window+j]);
		for (int j = 0; j < dimensionWPE * window; j++)
			fscanf(fin, "%f",&matrixW1PositionE2[idx][i* dimensionWPE*window+j]);
		fscanf(fin, "%f", &matrixB1[idx][i]);
		//cout<<matrixB1[idx][i]<<' ';
	}
	cout<<endl;
	fclose(fin);

	fin = fopen(("./out/matrixRl.txt"+version).c_str(), "r");
	fscanf(fin,"%d%d", &relationTotal, &dimensionC);
	for (int idx = 0; idx<num_language; idx++)
	{
		for (int i = 0; i < relationTotal; i++) {
			for (int j = 0; j < dimensionC; j++)
			{
				fscanf(fin, "%f", &matrixRelation[0][i * dimensionC + j]);
				fscanf(fin, "%f", &embRelation[idx][i * dimensionC + j]);
			}
		}
		for (int i = 0; i < relationTotal; i++) 
			fscanf(fin, "%f",&matrixRelationPr[0][i]);
	}
	fclose(fin);

	fin = fopen(("./out/matrixPosition.txt"+version).c_str(), "r");
	fscanf(fin,"%d%d%d", &PositionLength, &PositionLength, &dimensionWPE);
	for (int idx = 0; idx<num_language; idx++)
	for (int i = 0; i < PositionLength; i++) {
		for (int j = 0; j < dimensionWPE; j++)
			fscanf(fin, "%f", &positionVecE1[idx][i * dimensionWPE + j]);
	}
	for (int idx = 0; idx<num_language; idx++)
	for (int i = 0; i < PositionLength; i++) {
		for (int j = 0; j < dimensionWPE; j++)
			fscanf(fin, "%f", &positionVecE2[idx][i * dimensionWPE + j]);
	}
	fclose(fin);

	fin = fopen(("./out/word2vec.txt"+version).c_str(), "r");
	for (int idx = 0; idx<num_language; idx++)
	{
		fscanf(fin,"%d%d",&wordTotal[idx],&dimension);
		wordVec[idx].resize((wordTotal[idx]+1) * dimension);
		for (int i = 0; i < wordTotal[idx]; i++)
		{
			for (int j=0; j<dimension; j++)
				fscanf(fin,"%f",&wordVec[idx][i*dimension+j]);
		}
	}
	fclose(fin);
}

void preprocess()
{
	cout<<74<<endl;
	init_para();
	cout<<76<<endl;
	cout<<79<<endl;
	init("zh",1,istest);
	cout<<80<<endl;
	init("en",0,istest);
	cout<<85<<endl;
	read_para();
}

int main()
{
	num_language = 2;
	istest = 0;
	preprocess();
	test_rel();
	//test(0);
	return 0;
}
