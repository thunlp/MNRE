#ifndef INIT_H
#define INIT_H
#include <cstring>
#include <cstdlib>
#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <float.h>


#include "utils.h"

using namespace std;

string version = "2";

int istest = 0;

int output_model = 0;

int num_threads = 32;
int trainTimes = 15;
float alpha = 0.01;
float reduce = 0.98;
int tt,tt1;
int dimensionC = 200;//1000;
int dimensionWPE = 5;//25;
int dimension = 50;
int window = 3;
int limit = 30;
int PositionLength = 2*limit+1;
float Belt = 0.001;

int num_language = 1;

vector<vector<float> > matrixB1, matrixW1, matrixRelation, matrixRelationPr;
vector<vector<float> > matrixB1Dao, matrixW1Dao, matrixRelationDao, matrixRelationPrDao;

vector<vector<float> > embRelation, embRelationDao;

vector<vector<float> > wordVec;
vector<vector<float> >  wordVecDao;
vector<vector<float> >  positionVecE1, positionVecE2, matrixW1PositionE1, matrixW1PositionE2;
vector<vector<float> >  positionVecDaoE1, positionVecDaoE2, matrixW1PositionE1Dao, matrixW1PositionE2Dao;
vector<int> wordTotal;


double mx = 0;
int batch = 16;
int npoch;
int len;
float rate = 1;
FILE *logg;

int relationTotal;
map<string,int> relationMapping;

vector<map<string,int> > wordMapping;
vector<vector<string> > wordList;
vector<vector<vector<int> > > trainLists, trainPositionE1, trainPositionE2;
vector<vector<int> > trainheadList, traintailList, trainrelationList, trainLength;
vector<vector<vector<int> > > testLists, testPositionE1, testPositionE2;
vector<vector<int> > testheadList, testtailList, testrelationList, testLength;
vector<string> nam;

vector<map<string,vector<int> > > bags_train, bags_test;

char buffer[100000];


void init_para()
{
	cout<<"Begin_init_para"<<endl;
	FILE* f = fopen("../../data/relation2id.txt", "r");
	while (fscanf(f,"%s",buffer)==1) {
		int id;
		fscanf(f,"%d",&id);
		relationMapping[(string)(buffer)] = id;
		relationTotal++;
		nam.push_back((string)(buffer));
	}
	fclose(f);
	cout<<"relationTotal:\t"<<relationTotal<<endl;

	matrixB1.resize(num_language);
	matrixW1.resize(num_language);
	matrixRelation.resize(num_language);
	matrixRelationPr.resize(num_language);
	wordVec.resize(num_language);
	positionVecE1.resize(num_language);
	positionVecE2.resize(num_language);
	matrixW1PositionE1.resize(num_language);
	matrixW1PositionE2.resize(num_language);
	wordTotal.resize(num_language);

	wordList.resize(num_language);
	wordMapping.resize(num_language);

	bags_train.resize(num_language);
	bags_test.resize(num_language);

	trainheadList.resize(num_language);
	trainheadList.resize(num_language);
	traintailList.resize(num_language);
	trainrelationList.resize(num_language);
	trainLength.resize(num_language);
	trainLists.resize(num_language);
	trainPositionE1.resize(num_language);
	trainPositionE2.resize(num_language);


	testheadList.resize(num_language);
	testheadList.resize(num_language);
	testtailList.resize(num_language);
	testrelationList.resize(num_language);
	testLength.resize(num_language);
	testLists.resize(num_language);
	testPositionE1.resize(num_language);
	testPositionE2.resize(num_language);

	embRelation.resize(num_language);
	for (int idx1 = 0; idx1<num_language; idx1++)
		embRelation[idx1].resize(3 * dimensionC * relationTotal);



	float con = sqrt(6.0/(dimensionC+relationTotal));
	float con1 = sqrt(6.0/((dimensionWPE+dimension)*window));
	for (int idx = 0; idx<num_language; idx++)
	{
		matrixRelation[idx].resize(3 * dimensionC * relationTotal);
		matrixRelationPr[idx].resize(relationTotal); 
		positionVecE1[idx].resize(PositionLength * dimensionWPE); 
		positionVecE2[idx].resize(PositionLength * dimensionWPE); 


		matrixW1[idx].resize(dimensionC * dimension * window);
		matrixW1PositionE1[idx].resize(dimensionC * dimensionWPE * window);
		matrixW1PositionE2[idx].resize(dimensionC * dimensionWPE * window);
		matrixB1[idx].resize(3 * dimensionC);

		for (int i = 0; i < dimensionC; i++) {
			int last = i * window * dimension;
			for (int j = dimension * window - 1; j >=0; j--)
				matrixW1[idx][last + j] = getRandU(-con1, con1);
			last = i * window * dimensionWPE;
			for (int j = dimensionWPE * window - 1; j >=0; j--) {
				matrixW1PositionE1[idx][last + j] = getRandU(-con1, con1);
				matrixW1PositionE2[idx][last + j] = getRandU(-con1, con1);
			}
			for (int j=0; j<3; j++)
			matrixB1[idx][3 * i + j] = getRandU(-con1, con1);
		}

		for (int i = 0; i < relationTotal; i++) 
		{
			matrixRelationPr[idx][i] = getRandU(-con, con);				//add
			for (int j = 0; j < 3 * dimensionC; j++)
				matrixRelation[idx][i * 3 * dimensionC + j] = getRandU(-con, con);
		}

		for (int i = 0; i < PositionLength; i++)
			for (int j = 0; j < dimensionWPE; j++)
				positionVecE1[idx][i * dimensionWPE + j] = getRandU(-con1, con1);

		for (int i = 0; i < PositionLength; i++)
			for (int j = 0; j < dimensionWPE; j++)
				positionVecE2[idx][i * dimensionWPE + j] = getRandU(-con1, con1);
	}
	//embRelation = matrixRelation;

	cout<<"End_init_para"<<endl;
}


void read_data(int flag, string name, map<string,int> wordMapping, vector<int> &headList, vector<int> &tailList, vector<int> &relationList, vector<int> &Length,
		vector<vector<int> > &Lists, vector<vector<int> > &PositionE1, vector<vector<int> > &PositionE2, map<string,vector<int> > &bags)
{
	FILE* f = fopen(("../../data/"+name+".txt").c_str(), "r");
	while (fscanf(f,"%s",buffer)==1)  {
		string e1 = buffer;
		fscanf(f,"%s",buffer);
		string e2 = buffer;
		fscanf(f,"%s",buffer);
		string head_s = (string)(buffer);
		int head = wordMapping[(string)(buffer)];
		fscanf(f,"%s",buffer);
		int tail = wordMapping[(string)(buffer)];
		string tail_s = (string)(buffer);
		fscanf(f,"%s",buffer);
		if (flag)
			bags[e1+"\t"+e2+"\t"+(string)(buffer)].push_back(headList.size());
		else
			bags[e1+"\t"+e2].push_back(headList.size());
		int num = relationMapping[(string)(buffer)];
		//cout<<e1+"\t"+e2+"\t"+(string)(buffer)<<' '<<num<<endl;
		int len = 0, lefnum = 0, rignum = 0;
		vector<int> tmpp;
		while (fscanf(f,"%s", buffer)==1) {
			std::string con = buffer;
			if (con=="###END###") break;
			int gg = wordMapping[con];
			if (con == head_s) lefnum = len;
			if (con == tail_s) rignum = len;
			len++;
			tmpp.push_back(gg);
		}
		headList.push_back(head);
		tailList.push_back(tail);
		relationList.push_back(num);
		Length.push_back(len);
		vector<int> con,conl, conr;
		for (int i = 0; i < len; i++) {
			con.push_back(tmpp[i]);
			conl.push_back(lefnum - i);
			conr.push_back(rignum - i);
			if (conl[i] >= limit) conl[i] = limit;
			if (conr[i] >= limit) conr[i] = limit;
			if (conl[i] <= -limit) conl[i] = -limit;
			if (conr[i] <= -limit) conr[i] = -limit;
			conl[i]+=limit;
			conr[i]+=limit;
		}
		Lists.push_back(con);
		PositionE1.push_back(conl);
		PositionE2.push_back(conr);
	}
	fclose(f);
}

void init(string language, int idx, int istest) {
	cout<<"Begin read data of language:\t"<<language<<endl;
	FILE *f = fopen(("../../data/vec_"+language+".bin").c_str(), "rb");
	int x;
	fscanf(f, "%d", &x);
	wordTotal[idx] = x;
	fscanf(f, "%d", &x);
	assert(dimension ==x);
	cout<<"wordTotal=\t"<<wordTotal[idx]<<endl;
	cout<<"Word dimension=\t"<<dimension<<endl;
	wordVec[idx].resize((wordTotal[idx]+1) * dimension);
	wordList[idx].resize(wordTotal[idx]+1);
	wordList[idx][0] = "UNK";
	for (int i = 1; i <= wordTotal[idx]; i++) {
		//cout<<i<<endl;
		string name = "";
		while (1) {
			char ch = fgetc(f);
			if (feof(f) || ch == ' ') break;
			if (ch != '\n') name = name + ch;
		}
		long long last = i * dimension;
		float smp = 0;
		for (int a = 0; a < dimension; a++) {
			float f_tmp;
			fread(&f_tmp, sizeof(float), 1, f);
			wordVec[idx][a+last] = f_tmp;
			smp += f_tmp*f_tmp;
		}
		smp = sqrt(smp);
		for (int a = 0; a< dimension; a++)
			wordVec[idx][a+last] = wordVec[idx][a+last] / smp;
		wordMapping[idx][name] = i;
		wordList[idx][i] = name;
	}
	wordTotal[idx]+=1;
	fclose(f);
	
	read_data(1, "train_"+language, wordMapping[idx], trainheadList[idx], traintailList[idx], trainrelationList[idx], trainLength[idx], 
		trainLists[idx], trainPositionE1[idx], trainPositionE2[idx], bags_train[idx]);
	cout<<240<<endl;
	if (istest==1)
		read_data(0, "test_"+language, wordMapping[idx], testheadList[idx], testtailList[idx], testrelationList[idx], testLength[idx], 
			testLists[idx], testPositionE1[idx], testPositionE2[idx], bags_test[idx]);
	else
	if (istest==0)
		read_data(0, "valid_"+language, wordMapping[idx], testheadList[idx], testtailList[idx], testrelationList[idx], testLength[idx], 
			testLists[idx], testPositionE1[idx], testPositionE2[idx], bags_test[idx]);
	else
	if (istest==2)
		read_data(0, "test_out_"+language, wordMapping[idx], testheadList[idx], testtailList[idx], testrelationList[idx], testLength[idx], 
			testLists[idx], testPositionE1[idx], testPositionE2[idx], bags_test[idx]);
	cout<<"End read data of language:\t"<<language<<endl;
}




#endif

