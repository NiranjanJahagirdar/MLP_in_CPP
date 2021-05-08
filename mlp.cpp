#include <bits/stdc++.h>
#include<fstream>
using namespace std;


/*-----------------------data structures------------------------------------*/
struct neuron{
  vector<double> weights;
  double bias;
  double delta;
  double data;
  int len;
};

struct layer{
  vector<neuron> nodes;
  int size;
};

/*--------------------------------Global Variables-----------------------*/
double learning_rate=0.01;
int datalen=10,numclasses=2,numlayers=1;
string actfunc="sigmoid";
vector<int> layerArch;
vector<layer> arch;
vector<vector<double>> input;
vector<vector<double>> output;
int epochs=5000;
int inputdim=3,outputdim=2;
/*--------------------------------------initialization functions---------*/
void inputneuron(neuron *n,int len){
  n->len=len;
  for(int i=0;i<len;i++){
    double r = ((double) rand() / (RAND_MAX));
    n->weights.push_back(r);
  }
  n->bias = ((double) rand() / (RAND_MAX));
  n->delta = ((double) rand() / (RAND_MAX));
  n->data = ((double) rand() / (RAND_MAX));
 return ;
}

void initialiseLayr(int size,layer *l,int len){
  l->size=size;
  for(int i=0;i<size;i++){
    neuron n;
    inputneuron(&n,len);
    l->nodes.push_back(n);
  }
  return ;
}

/*-------------------------------Activation function------------------------------*/

double sigmoidx(double x){ //activation function sigmoid
  double ans=0.0;
  ans = 1/(1+exp(-x));
  return ans;
}

double sigmoidxD(double sig){ //derivative of sigmoid => sig(z)(1-sig(z))
  return sig*(1-sig);
}

double tanhx(double x){ //activation function tanh= (e^z - e^-z)/e^z + e^-z
  return tanh(x);
}

double tanhxD(double tan){ //derivative of tanh => 1-tanh(z)^2
  return 1-pow(tanh(tan),2);
}

double relux(double x){ // Relu Activation Function
  return max(0.0,x);
}

double reluxD(double x){//derivative of Relu
  if(x<0.0)return 0.0;
  return 1.0;
}

double elux(double x){
  if(x>1.0) return x;
  return (0.1*exp(x)-1);
}

double eluxD(double elu){
  if(elu>0.0) return 1.0;
  return elu+0.1;
}

double leakyrelux(double x){
  return max(0.1*x,x);
}

double leakyreluxD(double x){
  if(x<0.0) return 0.1;
  return 1;
}

double activationX(double x){
  if(actfunc=="sigmoid"){
    return sigmoidx(x);
  }else if(actfunc=="tanh"){
    return tanhx(x);
  }else if(actfunc=="relu"){
    return relux(x);
  }else if(actfunc=="leakyrelu"){
    return leakyrelux(x);
  }else return elux(x);
}


double activationD(double x){
  if(actfunc=="sigmoid"){
    return sigmoidxD(x);
  }else if(actfunc=="tanh"){
    return tanhxD(x);
  }else if(actfunc=="relu"){
    return reluxD(x);
  }else if(actfunc=="leakyrelu"){
    return leakyreluxD(x);
  }else return eluxD(x);
}
/*--------------------------MLP------------------------------------------------*/

vector<layer> MLP(vector<int> layerInfo){  //Creating Layer Architecture
  vector<layer> l;
  for(int i=0;i<layerInfo.size();i++){
    if(i!=0){
      layer k;
      initialiseLayr(layerInfo[i],&k,layerInfo[i-1]);
      l.push_back(k);
    }else{
      layer k;
      initialiseLayr(layerInfo[i],&k,0);
      l.push_back(k);       // Input Layer
    }
  }
  return l;
}

vector<double> loadoutput(){
  vector<double> output;
  int size=arch.size()-1;
  for(int i=0;i<arch[size].size;i++){
    output.push_back(arch[size].nodes[i].data);
  }
  return output;
}

vector<double> feedForward(vector<double> input){
  double out=0.0;
  for(int k=0;k<arch.size();k++){ //interations over all arch and updating values
    for(int i=0;i<arch[k].size;i++){
      if(k==0){
        for(int i=0;i<arch[0].size;i++){
          arch[0].nodes[i].data = input[i];   //First Layer is input Layer
        }
      }else{
      out=0.0;
      for(int j=0;j<arch[k-1].size;j++){
        double temp1=arch[k].nodes[i].weights[j];
        double temp2=arch[k-1].nodes[j].data;
        out+= temp1*temp2; // summation (WiXi)
      }
      out+=arch[k].nodes[i].bias ;
      arch[k].nodes[i].data = activationX(out);
        }
        out=0.0;
      }
  }
  return loadoutput();
}

double errorFunc(vector<double> pred,vector<double> output){
  double error=0.0;
  for(int i=0;i<output.size();i++){
    error+=abs(pred[i]-output[i]);
  }
  error=error/output.size();
  return error;
}

double backPropagation(vector<double> input,vector<double> output){
  double error;
  vector<double> pred = feedForward(input);
  int size1=arch.size();
  for(int i=0;i<arch[arch.size()-1].size;i++){
    error = output[i]-pred[i];
    arch[arch.size()-1].nodes[i].delta = error*activationD(pred[i]);   //deltata(W)=(desired-predicted)*derivative(sigmoid)*X
  }
  for(int k=arch.size()-2;k>=0;k--){
    for(int i=0;i<arch[k].size;i++){
      error=0.0;
      for(int j=0;j<arch[k+1].size;j++){
        error +=arch[k+1].nodes[j].delta* arch[k+1].nodes[j].weights[i]; //Dot Product W and deltata(w)
      }
      arch[k].nodes[i].delta = error*activationD(arch[k].nodes[i].data);
      error=0.0;
    }
    for(int i=0;i<arch[k+1].size;i++){
      for(int j=0;j<arch[k].size;j++){ //updating weights
        double temp4=learning_rate*arch[k+1].nodes[i].delta;
        temp4=temp4*arch[k].nodes[j].data;
        arch[k+1].nodes[i].weights[j]+=temp4;
      }
      double temp5=learning_rate;
      arch[k+1].nodes[i].bias += temp5*arch[k+1].nodes[i].delta;
    }
    error=0.0;
  }
  double ans=errorFunc(pred,output);
  return ans;
}
/*-----------------------------Print Functions------------------------------------*/

void printneuron(neuron n){
  cout<<"weights : ";
  for(int i=0;i<n.len;i++){
    cout<<n.weights[i]<<" , ";
  }
  cout<<endl;
  cout<< "bias : " <<n.bias<<endl;
  cout<< "delta : "<<n.delta<<endl;
  cout<< "data : "<<n.data<<endl;
  cout<< "len : " <<n.len<<endl;
  cout<<endl;
}

void printlayer(layer l){
  cout<<" layer "<< endl;
  for(int i=0;i<l.size;i++){
    cout<<"neuron "<<i+1<<endl;
    printneuron(l.nodes[i]);
  }
  return ;
}

void printlayerdim(){
  cout<<"\nPARAMETERS"<<endl;
  cout<<"Learning rate : "<<learning_rate<<endl;
  cout<<"Output Dimension : "<<numclasses<<endl;
  cout<<"Activation Function : "<<actfunc<<endl;
  cout<<"No of Input Features = "<<input[0].size()<<endl;
  cout<<"Size of Input Dataset = "<< input.size()<<endl;
  cout<<"\nModel Architecture = (";
  for(int i=0;i<layerArch.size();i++){
    if(i==layerArch.size()-1){
      cout<<layerArch[i]<<")";
    }else{
      cout<<layerArch[i]<<",";
    }
  }
  cout<<"\nNUMBER NEURONS IN EACH LAYER \n";

  for(int i=0;i<arch.size();i++){
    if(i==0){
      cout<<"INPUT LAYER   : "<<arch[i].size<<endl;
    }else if(i==arch.size()-1){
      cout<<"OUTPUT LAYER  : "<<arch[i].size<<endl;
    }else{
      cout<<"HIDDEN LAYER  : "<<arch[i].size<<endl;
    }
  }
  cout<<endl;
  return;
}
/*----------------------------------------supporting functions--------------------*/
vector<int> getparam(){
  vector<int> layr;
  cout<<"Enter Learning Rate"<<endl;
  cin>>learning_rate;
  cout<<"Enter Number of hidden Layers"<<endl;
  cin>>numlayers;
  cout<<"Enter No of neurons in Each Hidden Layer"<<endl;
  layr.push_back(3);
  int temp=0;
  for(int i=0;i<numlayers;i++){
    cin>>temp;
    layr.push_back(temp);
  }
  layr.push_back(2);
  cout<<"Enter Actication Function : sigmoid ,tanh, relu, leakyrelu ,elu ?"<<endl;
  cin>>actfunc;
  cout<<"Number of Epochs = "<<endl;
  cin>>epochs;
  return layr;
}

void inputdataset(string fname){
  ifstream fin;
  fin.open(fname);
  if(!fin){
    cout<<"File could not be accessed\n";
    exit(0);
  }
  //cout<<"Read file"<<endl;
  double in1,in2,in3,out;
  string line,word,temp;
  vector<string> row;
  while(fin >> temp){
    //cout<<temp<<endl;
    row.clear();
    //getline(fin,line);
    //cout<<line<<endl;
    stringstream s(temp);
    while(getline(s,word,',')){
      row.push_back(word);
    }
    in1=stod(row[0]);
    in2=stod(row[1]);
    in3=stod(row[2]);
    out=stod(row[3]);
    input.push_back({in1,in2,in3});
    if(out==1.0){
      output.push_back({1,0});
    }else{
      output.push_back({0,1});
    }
  }
  //cout<<"--------------------------------------------------------"<<endl;
/*
  for(int i=0;i<input.size();i++){
    for(int j=0;j<input[i].size();j++){
      cout<<input[i][j]<<" , ";
    }
    cout<<output[i];
    cout<<endl;
  }
  */
  /*for(int i=0;i<output.size();i++){
    for(int j=0;j<output[i].size();j++){
      cout<<output[i][j]<<" , ";
    }
    cout<<endl;
  }
*/
  return ;
}
/*-------------------------------------------main()----------------------------------*/
int main(){
  int trigger=0;
  cout<<"Training on HaberMan dataset\n";
  cout<<"To Run preloaded model press 1"<<endl<<"if you want to give your own inputs press 0"<<endl;
  cin>>trigger;
  //getparam();
  inputdataset("haberman.txt");
  double error=0.0;
  layerArch={3,5,5,5,2};
  if(trigger==0){
    layerArch = getparam();
  }
  arch=MLP(layerArch);
  printlayerdim();
  cout<<"Model Running..."<<endl;
  for(int iter=0;iter<=epochs;iter++){
    double count1=0;
    for(int j=0;j<input.size();j++){
      vector<double> invar=input[j];
      vector<double> outvar=output[j];
      error=backPropagation(invar,outvar);
      vector<double> predicted=feedForward(invar);
    }
    if(iter%1000==0 & iter!=0){
      cout<<"Error At Step "<<iter<<" : "<<error<<endl;
    //  cout<<"correct= "<<count1;
      //double acc=count1/input.size();
      //cout<<", Accuracy % = "<<acc*100<<endl;
    }

  }

  //Calculating Accuracy
  /*double count=0;
  for(int i=0;i<input.size();i++){
    vector<double> invar=input[i];
    vector<double> outvar=output[i];
    vector<double> predicted=feedForward(invar);
    if(predicted[0]>predicted[1] && outvar[0]==1.0) count++;
    else if(predicted[0]<predicted[1] && outvar[1]==1.0) count++;
  }
  cout<<"\nCount Of Correctly Predicted Output= "<<count<<endl;
  double acc=count/input.size();
  cout<<"Accuracy % = "<<acc*100<<endl;
*/
  /*
  for(int i=0;i<arch.size();i++){
    printlayer(arch[i]);
    cout<<endl;
  }
  */


  //initialiseLayr(2,&l,3);
  //printlayer(l);
return 0;
}
