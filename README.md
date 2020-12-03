# What is Driving Cancer Rates at the US Census Tract Level?

Alex Truby

Galvanize Capstone 2

-----
## Project Goal

Over the years, the issue of Environemental Justice has become one of particular interest to me. The EPA has put together an "Environmental Justice Screening Tool" (EJST), which utilizes both environmental and demographic indicators in an effort to help policy makers identify communites' potential risk resulting from lack of enforcement around environmental laws, regulations, and policies. The screening tool does not, however, *actually* provide an "index" on which communicaties can be evaluated and ranked. 

While cancer rates are by no means the only indicators of community health and well being, that is what I have chosen as my target for this capstone, as a proxy for areas of concern regarding environmental justice. 

#### My goal is to utilize similar features (both demographic and environmental) that the EPA uses as inputs into the EJST to better understand what most drives cancer rates in populations at the census tract level in the US.

To accomplish this goal, I've evaluted three different inferential models (linear regression, random forest, and gradient boost) to better understand what factors have the most influence on cancer rates. 

-----

## The Data

While my goal was to find data sets for all of the features that are utilized in the EJST, not all of them are publically available at the census tract level (many only get down to the county level). As a result, this model is a bit limited to proxy data sets that I was able to find for the census tract level. Additionally, cancer rates were only avaialble for 500 cities at the census tract level, so this study is limited to that scope.

The below table gives a snapshot of the data as it was initially collected. 

![Raw Data Summary Table]('./images/capstone_2_raw_data.png')

Each table had to be slightly modified in order to be able to combine them all into a single table, broken out at the census tract level. The script for that data cleaning process can be found in the src folder of this repo.

----

## Inital EDA

A quick scatterplot of my model inputs told me that many of the demographic feautures I wanted to use as inputs to my models were likely correlated. Additonally, a few of the chemical hazard indexes (HIs) I had pulled in only contained zero values. So, I dropped all of the HIs with zero values, and made a separate dataframe of only the demographic features to input into a PCA model to find out how many of them will actually be useful to my model. 

The results from the PCA on the demographic data indicated that I really only needed to include two of the original four variables to account for 90% of the variance (see below image). I took this into account when building out my models later in the process. While only two demographic variables were needed to explain 90% of the variance, I decided to keep a third after this first inital pass of the EDA to have 95% of the variance explained, understanding I may want to drop it later in the process.

![Demographic PCA Plot]('./images/dem_pca.png')

This first pass of EDA left me with 7 features as inputs into my models: 4 environmental, and 3 demographic.

Further EDA, particularly for the linear regression model, will be covered later in the README.

-----

## Models Tested

As mentioned earlier, the goal of this project is to understand *what* is driving cancer rates in these census tracts, so I chose to focus on inferential regression models. The models of focus for this project are listed below - they were chosen because they are each able to provide insight around which of the input variables is most influencing cancer rates. 

### Linear Regression
Linear regression models arguably offer the best insight as to which variables are influencing the target by way of the model coefficients. However, in order for the coefficients to be interprettable, the model must meet the following five assumptions:

* Linearity (relationship between X and y)
* Independence
* No multicollinearity between features
    * One way to check for multicollinearity between input features is to check the variance inflation factor (VIF). A general rule of thumb is that if the VIF of a feature is greater than 10, is it likely collinear with another input for the model
    * The following results were obtained after calculating the VIF for each of the seven input features:

    <div align="center">

    |Variable                | VIF     |
    | -------------          | ------- |
    |has_superfund           |  1.00   |
    |acetaldehyde_HI         | 14.65   |
    |diesel_HI               |  3.26   |
    |particulate_matter      | 19.71   |
    |percent_minority        | 9.22    |
    |percent_no_hs_diploma   | 4.73    |
    |percent_over_65_yrs     | 4.16    |
    |                        |         |


    * <div align="left">As can be seen above, there are a few features with a VIF above 10. Additionally, given what we learned from the earlier PCA, I felt pretty confident that the percent_minority could be dropped. It is right on the cusp of the VIF threshold, and the earlier PCA illustrated that likely only two demographic variables are needed. Dropping both the particulate_matter and percent_minority variables, and running the VIF test again, all input variables had a VIF below ten.

<div align="center">

|Variable                | VIF     |
| -------------          | ------- |
|has_superfund           |  1.00   |
|acetaldehyde_HI         |  6.91   |
|diesel_HI               |  3.22   |
|percent_no_hs_diploma   |  2.37   |
|percent_over_65_yrs     |  3.51   |
|                        |         |

<div align="left">


* Normally distributed residuals - This assumption was NOT met by the data. Further explanation and data exploration around where and why this failed can be found **HERE** (link to another README.)
* Variance of the residuals is constant - This assumption also was NOT met by the data. The link above contains a detailed exploration of this assumption as well. 

Because not all of the assumptions for lienar regression were met by the data, the coefficients from this model will not give accurate insight as to how the features are impacting the target. Given that obtaining this insight was the goal of this project, we will move on to gradient boost and random forest for the remainder of the anlaysis.

### Gradient Boost

### Random Forest

-----

## Final Model
Because the results of the Random Forest & Gradient Descent models were both extremely comprable in terms of RMSE, I'll focus on the Random Forest Model, because I feel that is more easily interpretable. 


-----

## Conclusion 