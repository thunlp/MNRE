#ifndef TEST_REL_H
#define TEST_REL_H
#include "init.h"
#include "test.h"
#include <algorithm>
#include <map>



map<int, vector<string> > c; 

map<int,pair<int,int> > result;


vector<double> test1(int idx, vector<int> &sentence, vector<int> &PositionE1, vector<int> &PositionE2, int len, vector<float> &max_pooling) {
	vector<double> r;
	r.resize(dimensionC);
	for (int i = 0; i < dimensionC; i++) {
		int last = i * dimension * window;
		int lastt = i * dimensionWPE * window;
		float mx = -FLT_MAX;
		int tmp = 0;
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
			if (res > mx) {
				mx = res;
				tmp = i1;
			}
		}
		tmp+=window/2;
		if (tmp>=0 && tmp<len)
			max_pooling[tmp]++;
		r[i] = mx + matrixB1[idx][i];
	}

	for (int i = 0; i < dimensionC; i++)
		r[i] = CalcTanh(r[i]);
	return r;
}
pthread_mutex_t mutex2;

void* test_rel_mode(void *id) 
{
	int int_id = (long long)id;
	for (int rel_id=relationTotal*int_id/num_threads+1; rel_id<min(relationTotal,relationTotal*(int_id+1)/num_threads+1); rel_id++)
	{
		//cout<<rel_id<<' '<<c[rel_id].size()<<endl;
		int tot_correct = 0;
		for (int ii = 0; ii < c[rel_id].size(); ii++)
		{
			vector<double> sum;
			vector<double> r_sum;
			r_sum.resize(dimensionC);
			for (int j = 0; j < relationTotal; j++)
				sum.push_back(0.0);
			map<int,int> ok;
			ok.clear();
			vector<vector<float> > weight_list;
			vector<vector<vector<float> > > max_pooling_list;
			for (int idx = 0; idx<num_language; idx++)
			{
				//cout<<29<<' '<<c[rel_id][ii]<<endl;
				int bags_size = bags_test[idx][c[rel_id][ii]].size();
				//cout<<31<<endl;
				if (bags_size==0)
					continue;
				//	cout<<idx<<' '<<bags_size<<' '<<b[ii]<<endl;
				//assert(bags_size!=0);

				vector<vector<double> > rList;
				vector<vector<float> > max_pooling_array;
				for (int k=0; k<bags_size; k++)
				{
					//cout<<k<<endl;
					int i = bags_test[idx][c[rel_id][ii]][k];
					vector<float> max_pooling;
					max_pooling.resize(testLength[idx][i]);
					ok[testrelationList[0][i]]=1;
					vector<double> r_tmp = test1(idx, testLists[idx][i],  testPositionE1[idx][i], testPositionE2[idx][i], testLength[idx][i], max_pooling);
					rList.push_back(r_tmp);
					max_pooling_array.push_back(max_pooling);
				}
				max_pooling_list.push_back(max_pooling_array);
			//	cout<<43<<endl;
				int sent_num = rList.size();
			//	for (int idx_att=idx; idx_att<idx+1; idx_att++)
				for (int idx_att=0; idx_att<num_language; idx_att++)
				{
					//if (idx!=1&&idx_att!=1)
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
						if (j==rel_id)
							weight_list.push_back(weight);
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
								s +=  0.5 * (embRelation[idx_att][j1 * dimensionC + i1] + 1 * matrixRelation[0][j1 * dimensionC + i1]) * r[i1];
							s += matrixRelationPr[0][j1];
							s = exp(s);
							tmp+=s;
							res.push_back(s);
						}
						sum[j] = sum[j] + res[j]/tmp;
					}
					//cout<<80<<endl;
				}
			}
			assert(ok[rel_id]==1);
			float x = sum[rel_id];
			int flag = 1;
			for (int j = 1; j < relationTotal; j++) 
				if (j!=rel_id&& ok[j]!=1 && sum[j]>x)
					flag = 0;

			pthread_mutex_lock (&mutex2);
			if (flag&&bags_test[0][c[rel_id][ii]].size()>=2&&bags_test[1][c[rel_id][ii]].size()>=2)
			{
				int jj = bags_test[0][c[rel_id][ii]][0];
				vector<int> flag1;
				flag1.resize(num_language);
				for (int idx = 0; idx<num_language; idx++)
				{
					int bags_size = bags_test[idx][c[rel_id][ii]].size();
					for (int i=idx*2; i<(idx+1)*2; i++)
						for (int k=0; k<bags_size; k++)
							if (weight_list[i][k]>0.6)
								flag1[idx] = 1;
				}
				if (flag1[0]==1&&flag1[1]==1)
				{
					cout<<c[rel_id][ii]<<' '<<wordList[0][testheadList[0][jj]]<<' '<<wordList[0][testtailList[0][jj]]<<' '<<nam[rel_id]<<endl;
					for (int idx = 0; idx<num_language; idx++)
					{
						int bags_size = bags_test[idx][c[rel_id][ii]].size();
						for (int k=0; k<bags_size; k++)
						{
							int i = bags_test[idx][c[rel_id][ii]][k];
							int len = testLength[idx][i];
							double sum = 0;
							for (int j=0; j<len; j++)
								sum+=max_pooling_list[idx][k][j];
							for (int j=0; j<len; j++)
								cout<<wordList[idx][testLists[idx][i][j]]<<' ';//<<max_pooling_list[idx][k][j]/sum<<' ';
							cout<<endl;
						}
						for (int i=idx*2; i<(idx+1)*2; i++)
						{
							for (int k=0; k<bags_size; k++)
								cout<<weight_list[i][k]<<' ';
							cout<<endl;
						}
						cout<<"-----------------------------"<<endl;
					}
				}
			}
			pthread_mutex_unlock(&mutex2);
			tot_correct+=flag;
		}
		result[rel_id] = make_pair(tot_correct, c[rel_id].size());
	}
}

void test_rel() {
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
		for (int i=1; i<relationTotal; i++)
			if (ok[i])
				c[i].push_back(it->first);
	}
	cout<<"tot = "<<tot<<' '<<num_threads<<endl;
	
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	for (int a = 0; a < num_threads; a++)
		pthread_create(&pt[a], NULL, test_rel_mode,  (void *)a);
	for (int a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);	
	cout<<136<<endl;
	for (int i=1; i<relationTotal; i++)
		cout<<result[i].first<<" "<<result[i].second<<endl;
}

#endif
