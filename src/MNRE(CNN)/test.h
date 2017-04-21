#ifndef TEST_H
#define TEST_H
#include "init.h"
#include <algorithm>
#include <map>

int tipp = 0;
float ress = 0;

vector<double> test(int idx, vector<int> &sentence, vector<int> &PositionE1, vector<int> &PositionE2, int len) {
	vector<double> r;
	r.resize(dimensionC);
	for (int i = 0; i < dimensionC; i++) {
		int last = i * dimension * window;
		int lastt = i * dimensionWPE * window;
		float mx = -FLT_MAX;
		for (int i1 = -window+1; i1 < len; i1++) {
			float res = 0;
			int tot = 0;
			int tot1 = 0;
			for (int j = i1; j < i1 + window; j++) 
			if (j>=0&&j<len)
			{
				int last1 = sentence[j] * dimension;
			 	for (int k = 0; k < dimension; k++) {
			 		res += matrixW1[idx][last + tot] * wordVec[idx][last1+k];
			 		tot++;
			 	}
			 	int last2 = PositionE1[j] * dimensionWPE;
			 	int last3 = PositionE2[j] * dimensionWPE;
			 	for (int k = 0; k < dimensionWPE; k++) {
			 		res += matrixW1PositionE1[idx][lastt + tot1] * positionVecE1[idx][last2+k];
			 		res += matrixW1PositionE2[idx][lastt + tot1] * positionVecE2[idx][last3+k];
			 		tot1++;
			 	}
			}
			if (res > mx) mx = res;
		}
		r[i] = mx + matrixB1[idx][i];
	}

	for (int i = 0; i < dimensionC; i++)
		r[i] = CalcTanh(r[i]);
	return r;
}


bool cmp(pair<string, pair<int,double> > a,pair<string, pair<int,double> >b)
{
    return a.second.second>b.second.second;
}

vector<string> b;
double tot;
vector<pair<string, pair<int,double> > >aa;

pthread_mutex_t mutex;
vector<int> ll_test;

void* testMode(void *id) 
{
	int ll = ll_test[(long long)id];
	int rr;
	if ((long long)id==num_threads-1)
		rr = b.size();
	else
		rr = ll_test[(long long)id+1];
	//cout<<ll<<' '<<rr<<' '<<((long long)id)<<endl;
	double eps = 0.1;
	for (int ii = ll; ii < rr; ii++)
	{
		vector<double> sum;
		vector<double> r_sum;
		r_sum.resize(dimensionC);
		for (int j = 0; j < relationTotal; j++)
			sum.push_back(0.0);
		map<int,int> ok;
		ok.clear();
		for (int idx = 0; idx<num_language; idx++)
		{
			int bags_size = bags_test[idx][b[ii]].size();
			if (bags_size==0)
				continue;
			//	cout<<idx<<' '<<bags_size<<' '<<b[ii]<<endl;
			//assert(bags_size!=0);

			vector<vector<double> > rList;
			for (int k=0; k<bags_size; k++)
			{
				int i = bags_test[idx][b[ii]][k];
				ok[testrelationList[0][i]]=1;
				vector<double> r_tmp = test(idx, testLists[idx][i],  testPositionE1[idx][i], testPositionE2[idx][i], testLength[idx][i]);
				rList.push_back(r_tmp);
			}
		
			int sent_num = rList.size();
		//	for (int idx_att=idx; idx_att<idx+1; idx_att++)
			for (int idx_att=0; idx_att<num_language; idx_att++)
			{
				//if (idx!=1&&idx_att!=1)
				//if (idx!=idx_att)
				//	continue;
				for (int j = 0; j < relationTotal; j++) {
					vector<float> weight;
					float weight_sum = 0;
					for (int k=0; k<sent_num; k++)
					{
						float s = 0;
						for (int i = 0; i < dimensionC; i++) 
							s += rList[k][i] * embRelation[idx_att][j * dimensionC + i];
						s = exp(s); 
						weight.push_back(s);
						weight_sum += s;
					}
					for (int k=0; k<sent_num; k++)
						weight[k] /=weight_sum;
					vector<float> r;
					r.resize(dimensionC);
					for (int i = 0; i < dimensionC; i++) 
						for (int k=0; k<sent_num; k++)
							r[i] += rList[k][i] * weight[k];
					vector<float> res;
					double tmp = 0;
					for (int j1 = 0; j1 < relationTotal; j1++) {
						float s = 0;
						for (int i1 = 0; i1 < dimensionC; i1++)
							s +=  0.5 * (0* embRelation[idx_att][j1 * dimensionC + i1] + 1 * matrixRelation[0][j1 * dimensionC + i1]) * r[i1];
						s += matrixRelationPr[0][j1];
						s = exp(s);
						tmp+=s;
						res.push_back(s);
					}
					sum[j] = sum[j] + res[j]/tmp;
				}
			}
		}
		pthread_mutex_lock (&mutex);
		for (int j = 1; j < relationTotal; j++) 
		{
			int i = bags_test[0][b[ii]][0];
			aa.push_back(make_pair(wordList[0][testheadList[0][i]]+' '+wordList[0][testtailList[0][i]]+' '+nam[j],make_pair(ok.count(j),sum[j])));
		}
		pthread_mutex_unlock(&mutex);
	}
}

void output_para(int turn)
{
	double correct=0;
	FILE* f = fopen(("out/pr"+version+".txt").c_str(), "w");

	fprintf(f,"%d\n",turn);
	for (int i=0; i<10000; i++)
	{
		//cout<<aa[i].second<<endl;
		if (aa[i].second.first!=0)
			correct++;	
		//if (i%100==1)
		//cout<<"precision:\t"<<correct/(i+1)<<'\t'<<"recall:\t"<<correct/tot<<endl;
		fprintf(f,"%lf\t%lf\t%lf\t%s\n",correct/(i+1), correct/tot,aa[i].second.second, aa[i].first.c_str());
	}
	fclose(f);
	if (!output_model)
		return;
	FILE *fout = fopen(("./out/matrixW1+B1.txt"+version).c_str(), "w");
	fprintf(fout,"%d\t%d\t%d\t%d\n", dimensionC, dimension, window, dimensionWPE);
	for (int idx = 0; idx<num_language; idx++)
	for (int i = 0; i < dimensionC; i++) {
		for (int j = 0; j < dimension * window; j++)
			fprintf(fout, "%f\t",matrixW1[idx][i* dimension*window+j]);
		for (int j = 0; j < dimensionWPE * window; j++)
			fprintf(fout, "%f\t",matrixW1PositionE1[idx][i* dimensionWPE*window+j]);
		for (int j = 0; j < dimensionWPE * window; j++)
			fprintf(fout, "%f\t",matrixW1PositionE2[idx][i* dimensionWPE*window+j]);
		fprintf(fout, "%f\n", matrixB1[idx][i]);
	}
	fclose(fout);

	fout = fopen(("./out/matrixRl.txt"+version).c_str(), "w");
	fprintf(fout,"%d\t%d\n", relationTotal, dimensionC);
	for (int idx = 0; idx<num_language; idx++)
	{
		for (int i = 0; i < relationTotal; i++) {
			for (int j = 0; j < dimensionC; j++)
			{
				fprintf(fout, "%f\t", matrixRelation[0][i * dimensionC + j]);
				fprintf(fout, "%f\t", embRelation[idx][i * dimensionC + j]);
			}
			fprintf(fout, "\n");
		}
		for (int i = 0; i < relationTotal; i++) 
			fprintf(fout, "%f\t",matrixRelationPr[0][i]);
		fprintf(fout, "\n");
	}
	fclose(fout);

	fout = fopen(("./out/matrixPosition.txt"+version).c_str(), "w");
	fprintf(fout,"%d\t%d\t%d\n", PositionLength, PositionLength, dimensionWPE);
	for (int idx = 0; idx<num_language; idx++)
	for (int i = 0; i < PositionLength; i++) {
		for (int j = 0; j < dimensionWPE; j++)
			fprintf(fout, "%f\t", positionVecE1[idx][i * dimensionWPE + j]);
		fprintf(fout, "\n");
	}
	for (int idx = 0; idx<num_language; idx++)
	for (int i = 0; i < PositionLength; i++) {
		for (int j = 0; j < dimensionWPE; j++)
			fprintf(fout, "%f\t", positionVecE2[idx][i * dimensionWPE + j]);
		fprintf(fout, "\n");
	}
	fclose(fout);

	fout = fopen(("./out/word2vec.txt"+version).c_str(), "w");
	for (int idx = 0; idx<num_language; idx++)
	{
		fprintf(fout,"%d\t%d\n",wordTotal[idx],dimension);
		for (int i = 0; i < wordTotal[idx]; i++)
		{
			for (int j=0; j<dimension; j++)
				fprintf(fout,"%f\t",wordVec[idx][i*dimension+j]);
			fprintf(fout,"\n");
		}
	}
	fclose(fout);
}

double max_pre = 0.0;

void test(int turn) {
	cout<<"version="<<version<<endl;
	aa.clear();
	b.clear();
	tot = 0;
	ll_test.clear();
	vector<int> b_sum;
	b_sum.clear();
	for (map<string,vector<int> >:: iterator it = bags_test[0].begin(); it!=bags_test[0].end(); it++)
	{
		
		map<int,int> ok;
		ok.clear();
		for (int k=0; k<it->second.size(); k++)
		{
			int i = it->second[k];
			if (testrelationList[0][i]>0)
				ok[testrelationList[0][i]]=1;
			//if (relationList[i]>0)
			//	ok[relationList[i]]=1;
		}
		tot+=ok.size();
	//	if (ok.size()>0)
		{
			b.push_back(it->first);
			b_sum.push_back(it->second.size());
		}
	}
	for (int i=1; i<b_sum.size(); i++)
		b_sum[i] += b_sum[i-1];
	int now = 0;
	ll_test.resize(num_threads);
	for (int i=0; i<b_sum.size(); i++)
		if (b_sum[i]>=b_sum[b_sum.size()-1]/num_threads*now)
		{
			ll_test[now] = i;
			now+=1;
		}
	cout<<"tot:\t"<<tot<<' '<<bags_test[0].size()<<endl;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	for (int a = 0; a < num_threads; a++)
		pthread_create(&pt[a], NULL, testMode,  (void *)a);
	for (int a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);
	//cout<<"begin sort"<<endl;
	cout<<aa.size()<<endl;
	sort(aa.begin(),aa.end(),cmp);
	double correct=0;
	float correct1 = 0;
	int output_flag = 0;
	float sum_pre = 0;
	for (int i=0; i<min(10000,int(aa.size())); i++)
	{
		if (aa[i].second.first!=0)
			correct1++;	
		float precision = correct1/(i+1);
		float recall = correct1/tot;
		if (i%500==0)
			cout<<"precision:\t"<<correct1/(i+1)<<'\t'<<"recall:\t"<<correct1/tot<<endl;
		if (i%1000==999)
		{
			sum_pre += precision;
		}
	}
	if (sum_pre>max_pre)
	{
		max_pre = sum_pre;
		output_flag = 1;
	}
	cout<<output_flag<<endl;
	if (output_flag)
		output_para(turn);
}

#endif
