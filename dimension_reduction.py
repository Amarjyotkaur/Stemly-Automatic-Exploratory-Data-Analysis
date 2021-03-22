# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA


class DimensionReduction():
    """
    
    
    Dimension reduction.
        Reduced feature dimensions. 
    Args:
        model_name (str): declare dimension reduction approach from {'low_variance_filter','high_correlation_filter', 'factor_analysis', 'principal_component_analysis'}.
        threshold (float): threshold for high correlation filter.
        n_components: declare number of principal components.
        
        
    """
    def __init__(self, model_name, threshold, n_components):
        self.input_dim = None
        self.output_dim = None
        self.model_name = model_name
        self.threshold = threshold 
        self.n_components = n_components 

    def fit(self, x):
        self.x = x
        self.input_dim = x.shape[1]

    def transform(self):
        if self.model_name == 'low_variance_filter':
            output = self.lvf()
        elif self.model_name == 'high_correlation_filter':
            output = self.hcf()
        elif self.model_name == 'factor_analysis':
            output = self.fa()
        elif self.model_name == 'principal_component_analysis':
            output = self.pca()
        self.output_dim = output.shape[1]
        return output

    def lvf(self):
        feature_df = pd.DataFrame(self.x)
        var = feature_df.var()
    
        keep_dim_index = []
        for i in range(0,len(var)):
            if var[i]>0:   #setting the threshold as 10%
                keep_dim_index.append(i)
        feature_df_lvf = (feature_df.loc[:,keep_dim_index]).copy(deep = True)
        importance = pd.DataFrame({'Feature':feature_df_lvf.columns,'Variance':feature_df_lvf.var()})
        importance = importance.reset_index()
        importance = importance.drop("index", axis=1)
        importance = importance.sort_values(by=['Variance'], ascending=False,ignore_index=True)
        print('Feature Importance by Var:','\n',importance)
        return feature_df_lvf

        

    def hcf(self):
        feature_df = pd.DataFrame(self.x)
        corr_matrix = feature_df.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        feature_df_hcf = feature_df.drop(to_drop, axis = 1).copy(deep = True)
        return feature_df_hcf

    def fa(self):
        feature_df = pd.DataFrame(self.x)
        factor = FactorAnalysis(n_components=self.n_components).fit(feature_df.values)
        loadings = pd.DataFrame(factor.components_.T,columns=['f1','f2','f3'])
        loadings = loadings.reset_index()
        loadings =loadings.rename(columns={'index':'Feature'})
        print('Loading Scores:','\n',loadings)
        
        #Calculate explained variance ratio 
        m = factor.components_
        n = factor.noise_variance_
        m1 = m**2
        m2 = np.sum(m1,axis=1)
        pvar1 = (m2[0])/np.sum(m2)
        pvar2 = (m2[1])/np.sum(m2)
        pvar3 = (m2[2])/np.sum(m2)
        varls= [pvar1,pvar2,pvar3]
        print("Explained Variance Ratio:",varls)
        #Calculate explained variance ratio including noise variance
        pvar1_with_noise = (m2[0])/(np.sum(m2)+np.sum(n))
        pvar2_with_noise = (m2[1])/(np.sum(m2)+np.sum(n))
        pvar3_with_noise = (m2[2])/(np.sum(m2)+np.sum(n))
        var_noise = [pvar1_with_noise, pvar2_with_noise, pvar3_with_noise]
        print("Exaplined Variance Ratio with NOISE:", var_noise,'\n')
        
        #Calculate weighted score
        for col in loadings.columns:
            loadings[col]=loadings[col].abs()
        ls=[]    
        for i in loadings["Feature"]:
            score = (loadings['f1'][i]*var_noise[0])+(loadings['f2'][i]*var_noise[1])+(loadings['f3'][i]*var_noise[2])
            ls.append(score)     
        importance = pd.DataFrame({'Feature':loadings["Feature"],'Score':pd.Series(ls)})
        importance = importance.sort_values(by =['Score'], ascending=False, ignore_index=True)
        print("Importance Scores: ","\n",importance)
        
        feature_df_fa = FactorAnalysis(n_components=self.n_components).fit_transform(feature_df.values)
        return feature_df_fa


    def pca(self):
        feature_df = pd.DataFrame(self.x)
        pca = PCA(self.n_components)
        # pca.fit_transform return pc1,...pcn
        # pc1 means dist(data points, origin) in pc1 axis
        feature_df_pca = pca.fit_transform(feature_df.values)
        print("explained variance ratio: ",pca.explained_variance_ratio_,'\n')
        # Cumulative explained variance
        print("cumulative explained var:",np.cumsum(pca.explained_variance_ratio_),'\n')

        loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2','PC3'])
        loadings = loadings.reset_index()
        loadings =loadings.rename(columns={'index':'Feature'})
        for col in loadings.columns:
            loadings[col]=loadings[col].abs()
        print(loadings)

        ls = []
        for i in loadings["Feature"]:
            score = (loadings['PC1'][i]*pca.explained_variance_ratio_[0])+(loadings['PC2'][i]*pca.explained_variance_ratio_[1])+(loadings['PC3'][i]*pca.explained_variance_ratio_[2])
            ls.append(score)
        importance = pd.DataFrame({'Feature':loadings["Feature"],'Score':pd.Series(ls)})
        importance = importance.sort_values(by =['Score'], ascending=False, ignore_index=True)
        print("Importance Scores: ","\n",importance)
        return feature_df_pca