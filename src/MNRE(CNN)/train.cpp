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

using namespace std;



float score = 0;
float alpha1;

float gradient_min = -1, gradient_max = 1;

int turn = 0;




vector<float> train(int idx, vector<int> &sentence, vector<int> &trainPositionE1, vector<int> &trainPositionE2, int len, vector<int> &tip) {
	vector<float> r;
	r.resize(dimensionC);
	//cout<<32<<endl;
	for (int i = 0; i < dimensionC; i++) {
		r[i] = 0;
		int last = i * dimension * window;
		int lastt = i * dimensionWPE * window;
		float mx = -FLT_MAX;
		for (int i1 = -window+1; i1 <len; i1++) {
			float res = 0;
			int tot = 0;
			int tot1 = 0;
			for (int j = i1; j < i1 + window; j++)  
			if (j>=0&&j<len)
			{
				//cout<<43<<' '<<idx<<' '<<sentence.size()<<' '<<matrixW1Dao[idx].size()<<' '<<matrixW1[idx].size()<<' '<<wordVecDao[idx].size()<<endl;
				int last1 = sentence[j] * dimension;
			 	for (int k = 0; k < dimension; k++) {
			 		res += matrixW1Dao[idx][last + tot] * wordVecDao[idx][last1+k];
			 		tot++;
			 	}
			 	//cout<<48<<endl;
			 	int last2 = trainPositionE1[j] * dimensionWPE;
			 	int last3 = trainPositionE2[j] * dimensionWPE;
			 	for (int k = 0; k < dimensionWPE; k++) {
			 		res += matrixW1PositionE1Dao[idx][lastt + tot1] * positionVecDaoE1[idx][last2+k];
			 		res += matrixW1PositionE2Dao[idx][lastt + tot1] * positionVecDaoE2[idx][last3+k];
			 		tot1++;
			 	}
			}
			if (res!=res)
			{
				float tmp = 0;
				for (int j = i1; j < i1 + window; j++)  
				if (j>=0&&j<len)
				{
					int last1 = sentence[j] * dimension;
				 	for (int k = 0; k < dimension; k++) 
				 		tmp += matrixW1Dao[idx][last + tot] * wordVecDao[idx][last1+k];
				 	cout<<73<<' '<<tmp<<endl;
				 	int last2 = trainPositionE1[j] * dimensionWPE;
				 	int last3 = trainPositionE2[j] * dimensionWPE;
				 	for (int k = 0; k < dimensionWPE; k++) 
				 	{
				 		tmp += matrixW1PositionE1Dao[idx][lastt + tot1] * positionVecDaoE1[idx][last2+k];
				 		tmp += matrixW1PositionE2Dao[idx][lastt + tot1] * positionVecDaoE2[idx][last3+k];
				 	}
				 	cout<<80<<' '<<tmp<<endl;
				}	
			}
			if (res > mx) {
				mx = res;
				tip[i] = i1;
			}
		}
		r[i] = mx + matrixB1Dao[idx][i];
		if (r[i]!=r[i])
		{
			cout<<91<<' '<<mx<<' '<<matrixB1Dao[idx][i]<<endl;
		}
	}

	for (int i = 0; i < dimensionC; i++) {
		r[i] = CalcTanh(r[i]);
		assert(r[i]==r[i]);
	}
	return r;
}

void train_gradient(int idx, vector<int> &sentence, vector<int> &trainPositionE1, vector<int> &trainPositionE2, int len, int e1, int e2, int r1, float alpha, vector<float> &r,vector<int> &tip, vector<float> &grad)
{
	for (int i = 0; i < dimensionC; i++) {
		if (fabs(grad[i])<1e-8)
			continue;
		int last = i * dimension * window;
		int tot = 0;
		int lastt = i * dimensionWPE * window;
		int tot1 = 0;
		float g1 = grad[i] * (1 -  r[i] * r[i]);
		for (int j = 0; j < window; j++) 
			if (tip[i]+j>=0&&tip[i]+j<len)
			{
				int last1 = sentence[tip[i] + j] * dimension;
				for (int k = 0; k < dimension; k++) {
					matrixW1[idx][last + tot] -= g1 * wordVecDao[idx][last1+k];
				//	assert(matrixW1[idx][last + tot]==matrixW1[idx][last + tot]);
					wordVec[idx][last1 + k] -= g1 * matrixW1Dao[idx][last + tot];
				//	assert(wordVec[idx][last1 + k]==wordVec[idx][last1 + k]);
					tot++;
				}
				int last2 = trainPositionE1[tip[i] + j] * dimensionWPE;
				int last3 = trainPositionE2[tip[i] + j] * dimensionWPE;
				for (int k = 0; k < dimensionWPE; k++) {
					matrixW1PositionE1[idx][lastt + tot1] -= g1 * positionVecDaoE1[idx][last2 + k];
				//	assert(matrixW1PositionE1[idx][lastt + tot1] ==matrixW1PositionE1[idx][lastt + tot1] );
					matrixW1PositionE2[idx][lastt + tot1] -= g1 * positionVecDaoE2[idx][last3 + k];
				//	assert(matrixW1PositionE2[idx][lastt + tot1]==matrixW1PositionE2[idx][lastt + tot1]);
					positionVecE1[idx][last2 + k] -= g1 * matrixW1PositionE1Dao[idx][lastt + tot1];
				//	assert(positionVecE1[idx][last2 + k]==positionVecE1[idx][last2 + k]);
					positionVecE2[idx][last3 + k] -= g1 * matrixW1PositionE2Dao[idx][lastt + tot1];
				//	assert(positionVecE2[idx][last3 + k]==positionVecE2[idx][last3 + k]);
					tot1++;
				}
			}
		matrixB1[idx][i] -= g1;
	}
}


pthread_mutex_t mutex2;

float train_bags(int idx, string bags_name)
{
//	cout<<102<<' '<<bags_name<<' '<<bags_train[0][bags_name].size()<<endl;
	vector<int> idx_list;
	vector<int> iList;
	vector<vector<float> > rList;
	rList.clear();
	vector<vector<int> > tipList;
	int r1 = -1;
	int bags_size = bags_train[idx][bags_name].size();
	for (int k=0; k<bags_size; k++)
	{
		idx_list.push_back(idx);
		vector<int> tip;
		tip.resize(dimensionC);
		int i = bags_train[idx][bags_name][k];
		iList.push_back(i);
		if (r1==-1)
			r1 = trainrelationList[idx][i];
		else
			assert(r1==trainrelationList[idx][i]);
		//cout<<k<<endl;
		rList.push_back(train(idx, trainLists[idx][i], trainPositionE1[idx][i], trainPositionE2[idx][i], trainLength[idx][i], tip));
		for (int i = 0; i < dimensionC; i++) 
			assert(rList[k][i]==rList[k][i]);
		tipList.push_back(tip);
	}
	int sent_num = rList.size();

	for (int k=0; k<sent_num; k++)
		for (int i = 0; i < dimensionC; i++) 
			assert(rList[k][i]==rList[k][i]);

	//for (int idx_att = 0; idx_att<num_language; idx_att++)
	
	
	vector<int> dropout;
	for (int i = 0; i < dimensionC; i++) 
		//dropout.push_back(1);
		dropout.push_back(rand()%2);

	vector<vector<float> > grad;
	grad.resize(sent_num);
	for (int k=0; k<sent_num; k++)
		grad[k].resize(dimensionC);
	float rt = 0.0;

	//for (int idx_att = idx; idx_att<idx+1; idx_att++)
	for (int idx_att = 0; idx_att<num_language; idx_att++)
	{
		//if (idx==&&idx_att==0)
		//if (idx!=idx_att)
		//	continue;
		pthread_mutex_lock (&mutex2);
		vector<float> weight;
		float weight_sum = 0;
		for (int k=0; k<sent_num; k++)
		{
			float s = 0;
			for (int i = 0; i < dimensionC; i++) 
				s += rList[k][i] * embRelationDao[idx_att][r1 * dimensionC + i];
			s = exp(s); 
			weight.push_back(s);
			weight_sum += s;
		}

		for (int k=0; k<sent_num; k++)
			for (int i = 0; i < dimensionC; i++) 
				assert(rList[k][i]==rList[k][i]);
		for (int k=0; k<sent_num; k++)
			weight[k]/=weight_sum;
		vector<float> f_r, f_r1;	
		float sum = 0, sum1 = 0;
		vector<float> r;
			r.resize(dimensionC);
			for (int i = 0; i < dimensionC; i++) 
				for (int k=0; k<sent_num; k++)
					r[i] += rList[k][i] * weight[k];
		for (int k=0; k<sent_num; k++)
			for (int i = 0; i < dimensionC; i++) 
				assert(rList[k][i]==rList[k][i]);
		for (int j = 0; j < relationTotal; j++) {	
			
			for (int k=0; k<sent_num; k++)
			for (int i = 0; i < dimensionC; i++) 
			{
				if (rList[k][i]!=rList[k][i])
					cout<<"224"<<' '<<j<<endl;
				assert(rList[k][i]==rList[k][i]);
			}
			float ss = 0, ss1=0;
			for (int i = 0; i < dimensionC; i++) {
				//ss += dropout[i] * r[i] * matrixRelationDao[0][j * dimensionC + i];
				//ss1 += dropout[i] * r[i] * embRelationDao[idx_att][j * dimensionC + i];
				ss += dropout[i] * r[i] * embRelationDao[idx_att][j * dimensionC + i];
			}
			ss += matrixRelationPrDao[0][j];
			f_r.push_back(exp(ss));
			sum+=f_r[j];
			f_r1.push_back(exp(ss1));
			sum1+=f_r1[j];
			//assert(f_r[j]<=2000);
		//	cout<<f_r[j]<<' '<<f_r1[j]<<' '<<sum<<' '<<sum1<<endl;
		}


		for (int k=0; k<sent_num; k++)
			for (int i = 0; i < dimensionC; i++) 
				assert(rList[k][i]==rList[k][i]);
	
		rt += log(f_r[r1]) - log(sum);
		if (!(rt==rt))
		{
			for (int k=0; k<sent_num; k++)
			for (int i = 0; i < dimensionC; i++) 
				assert(rList[k][i]==rList[k][i]);
			cout<<rt<<' '<<f_r[r1]<<' '<<sum<<' '<<sent_num<<endl;
		//	for (int i=0; i<dimensionC; i++)
		//		cout<<matrixRelationDao[0][0*dimensionC+i]<<' ';
		//	cout<<endl;

			for (int i=0; i<relationTotal; i++)
				cout<<f_r[i]<<' ';
			cout<<endl;
			for (int k=0; k<sent_num; k++)
			for (int i = 0; i < dimensionC; i++) 
				assert(rList[k][i]==rList[k][i]);
			for (int k=0; k<sent_num; k++)
			{
				for (int i = 0; i < dimensionC; i++) 
					cout<<rList[k][i]<<' '<<weight[k]<<' '<<(*(unsigned int*)(&rList[k][i]))<<' '<<(rList[k][i]==rList[k][i])<<' ';
				cout<<endl;
			}
			assert(rt==rt);
		}
		//for (int j = 0; j < relationTotal; j++) 
		//	for (int i = 0; i < dimensionC; i++)
		//		 matrixRelationDao[0][j * dimensionC + i]+=embRelationDao[0][j * dimensionC + i];
		pthread_mutex_unlock (&mutex2);

	//	cout<<165<<' '<<sent_num<<endl;
		vector<float> g1_tmp;
		g1_tmp.resize(dimensionC);
		for (int r2 = 0; r2<relationTotal; r2++)
		{	
			vector<float> r;
			r.resize(dimensionC);
			for (int i = 0; i < dimensionC; i++) 
				for (int k=0; k<sent_num; k++)
					r[i] += rList[k][i] * weight[k];
			
			float g = f_r[r2]/sum*alpha1;
			if (r2 == r1)
				g -= alpha1;
			float gg = f_r1[r2]/sum1*alpha1;
			if (r2 == r1)
				gg -= alpha1;
			for (int i = 0; i < dimensionC; i++) 
			{
				float g1 = 0;
				if (dropout[i]!=0)
				{
					//g1 += g * matrixRelationDao[0][r2 * dimensionC + i];
					//matrixRelation[0][r2 * dimensionC + i] -= g * r[i];
				//	if (turn<8)
					embRelation[idx_att][r2 * dimensionC + i] -= g * r[i];
					g1 += g * embRelationDao[idx_att][r2 * dimensionC + i];
					//g1 += gg *  embRelationDao[idx_att][r2 * dimensionC + i];
				}
				g1_tmp[i]+=g1;
			}
			matrixRelationPr[0][r2] -= g;
			assert(matrixRelationPr[0][r2]==matrixRelationPr[0][r2]);
		}
	//	cout<<197<<endl;
		for (int i = 0; i < dimensionC; i++) 
		{
			float g1 = g1_tmp[i];
			float tmp_sum = 0; //for rList[k][i]*weight[k]
			for (int k=0; k<sent_num; k++)
			{
				grad[k][i]+=g1*weight[k];
				grad[k][i]+=g1*rList[k][i]*weight[k]*embRelationDao[idx_att][r1 * dimensionC + i];
				embRelation[idx_att][r1 * dimensionC + i]  += g1*rList[k][i]*weight[k]*rList[k][i];
				assert(embRelation[idx_att][r1 * dimensionC + i]==embRelation[idx_att][r1 * dimensionC + i]);
				tmp_sum += rList[k][i]*weight[k];
			}	
			for (int k1=0; k1<sent_num; k1++)
			{
				grad[k1][i]-=g1*tmp_sum*weight[k1]*embRelationDao[idx_att][r1 * dimensionC + i];
				embRelation[idx_att][r1 * dimensionC + i] -= g1*tmp_sum*weight[k1]*rList[k1][i];
				assert(embRelation[idx_att][r1 * dimensionC + i]==embRelation[idx_att][r1 * dimensionC + i]);
			}
		}

		//for (int j = 0; j < relationTotal; j++) 
		//	for (int i = 0; i < dimensionC; i++)
		//		 matrixRelationDao[0][j * dimensionC + i]-=embRelationDao[0][j * dimensionC + i];
		
	}
	//cout<<215<<endl;
	for (int k=0; k<sent_num; k++)
	{
		//cout<<k<<endl;
		int idx = idx_list[k];
		int i = iList[k];
		//cout<<"\t"<<k<<' '<<i<<' '<<bags_name<<endl;
		train_gradient(idx, trainLists[idx][i], trainPositionE1[idx][i], trainPositionE2[idx][i], trainLength[idx][i], trainheadList[idx][i], traintailList[idx][i], trainrelationList[idx][i], alpha1,rList[k], tipList[k], grad[k]);
		//cout<<k<<endl;
	}
//	cout<<221<<endl;
	return rt;
}


int test_tmp = 0;

vector<string> b_train;
vector<int> c_train;
float score_tmp = 0, score_max = 0;
pthread_mutex_t mutex1;

int tot_batch;
void* trainMode(void *id ) {
		unsigned long long next_random = (long long)id;
		test_tmp = 0;
	//	for (int k1 = batch; k1 > 0; k1--)
		while (true)
		{

			pthread_mutex_lock (&mutex1);
			if (score_tmp>=score_max)
			{
				pthread_mutex_unlock (&mutex1);
				break;
			}
			score_tmp+=1;
		//	cout<<score_tmp<<' '<<score_max<<endl;
			pthread_mutex_unlock (&mutex1);
			int j = getRand(0, c_train.size());
			//cout<<j<<'|';
			j = c_train[j];
			for (int idx = 0; idx<num_language; idx++)
				if (bags_train[idx][b_train[j]].size()>0)
					score += train_bags(idx, b_train[j]);
			//cout<<"257"<<endl;
		}
		//cout<<259<<endl;
		//cout<<endl;
}

void train() {
	//cout<<252<<endl;

	int tmp = 0;
	b_train.clear();
	c_train.clear();

	for (int idx = 0; idx<num_language; idx++)
	for (map<string,vector<int> >:: iterator it = bags_train[idx].begin(); it!=bags_train[idx].end(); it++)
	{
		int max_size = 1;//it->second.size()/2;
		for (int i=0; i<max(1,max_size); i++)
			c_train.push_back(b_train.size());
		b_train.push_back(it->first);
		tmp+=it->second.size();

		//if (c_train.size()>200000)
		//	break;
	}
	cout<<c_train.size()<<endl;

	//cout<<266<<endl;
	//time_begin();
//	test(0);
	//time_end();
//	return;
	for (turn = 0; turn < trainTimes; turn ++) {

	//	len = trainLists.size();
		len = c_train.size();
		npoch  =  len / (batch * num_threads);
		alpha1 = alpha*rate/batch;

		score = 0;
		score_max = 0;
		score_tmp = 0;
		float score1 = score;
		time_begin();
		for (int k = 1; k <= npoch; k++) {
			score_max += batch * num_threads;
			//cout<<k<<endl;
			positionVecDaoE1 = positionVecE1;
			positionVecDaoE2 = positionVecE2;
			matrixW1PositionE1Dao = matrixW1PositionE1;
			matrixW1PositionE2Dao = matrixW1PositionE2;
			wordVecDao = wordVec;

			matrixW1Dao = matrixW1;
			matrixB1Dao = matrixB1;
			matrixRelationPrDao = matrixRelationPr;
			matrixRelationDao = matrixRelation;
			embRelationDao = embRelation;

			pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
			for (int a = 0; a < num_threads; a++)
				pthread_create(&pt[a], NULL, trainMode,  (void *)a);
			for (int a = 0; a < num_threads; a++)
			pthread_join(pt[a], NULL);
			free(pt);
			if (k%(npoch/10)==0)
			{
				cout<<"npoch:\t"<<k<<'/'<<npoch<<endl;
				time_end();
				time_begin();
				cout<<"score:\t"<<score-score1<<' '<<score_tmp<<endl;
				score1 = score;
			}

		}
		printf("Total Score:\t%f\n",score);
		printf("test\n");
		test(turn);
		//if ((turn+1)%1==0) 
		//	rate=rate*0.9;
	}
	cout<<"Train End"<<endl;
}



int main(int argc, char ** argv) {
	num_language = 2;
	istest = 0;
	srand(1);
	for(int i=0;i<10;i++){ 
		int ran_num=rand() % 6;
		cout<<ran_num<<" ";
	}
	output_model = 1;
	logg = fopen("log.txt","w");
	cout<<"Init Begin."<<endl;
	init_para();
	init("zh",1,istest);
	//init("en",1);
	init("en",0,istest);
	//for (map<string,vector<int> >:: iterator it = bags_train.begin(); it!=bags_train.end(); it++)
	//	cout<<it->first<<endl;
	cout<<bags_train[0].size()<<' '<<bags_test[0].size()<<endl;
	//cout<<bags_train[1].size()<<' '<<bags_test[1].size()<<endl;
	cout<<"Init End."<<endl;
	train();
	fclose(logg);
}
