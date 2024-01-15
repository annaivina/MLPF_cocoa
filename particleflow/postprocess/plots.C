#include <TH2.h>
#include <TStyle.h>


#include<cmath>
#include<cassert>
#include<iostream>


#include"TFile.h"
#include"TH1.h"
#include"TH2.h"
#include"TCanvas.h"
#include"TAxis.h"
#include"TROOT.h"
#include"TF1.h"
#include"TMath.h"
#include"TStyle.h"
#include"TBranch.h"
#include"TTree.h"
#include"TColor.h"
#include"TLegend.h"
#include"TArray.h"
#include "TFitResult.h"
#include "TMatrixDSym.h"
#include "TRandom.h"
#include "TLine.h"
#include "TPad.h"
#include "TEfficiency.h"
#include "TDirectory.h"
#include<string>
#include "TString.h"
#include "TGraphAsymmErrors.h"
#include "AtlasLabels.C"
#include "AtlasStyle.C"
#include "TGraph.h"
#include "THStack.h"



void plots(){


	gStyle->Reset();
    SetAtlasStyle();
    gStyle->SetPalette(55);

    TFile *files  =  new TFile("test.root");
    TTree *jets = (TTree*)files->Get("jets");

    std::vector<float> *gen_jet_pt = nullptr;
    std::vector<float> *pred_jet_pt = nullptr;
    jets->SetBranchAddress("gen_jet_pt", &gen_jet_pt);
    jets->SetBranchAddress("pred_jet_pt", &pred_jet_pt);


    TH1F *LeadJdiff = new TH1F("LeadJdiff", "PT Difference", 130, -30, 100);

    for(int i =0; i< jets->GetEntries();++i){
    	jets->GetEntry();

    	float lead_gen;
    	float lead_pred;
    	if (!gen_jet_pt->empty()) {
            std::sort(gen_jet_pt->begin(), gen_jet_pt->end(), std::greater<float>());
            lead_gen = gen_jet_pt->at(0); // Fill with the highest pt
        }

        if (!pred_jet_pt->empty()) {
            std::sort(pred_jet_pt->begin(), pred_jet_pt->end(), std::greater<float>());
            lead_pred = pred_jet_pt->at(0); // Fill with the highest pt
        }

        float diff = lead_gen - lead_pred;

        LeadJdiff->Fill(diff);


    }

    TCanvas *canvas= new TCanvas("canvas","canvas",800,600);
    canvas->cd();


    LeadJdiff->SetLineColor(kRed);
    LeadJdiff->SetMarkerColor(kRed);
    LeadJdiff->SetMarkerStyle(20);
    LeadJdiff->SetTitle(";diff; Events");
    LeadJdiff->Draw("hist");









}
