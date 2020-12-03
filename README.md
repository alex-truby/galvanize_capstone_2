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

----


## EDA

-----

## Models Tested

### Linear Regression


### Gradient Boost

### Random Forest

-----

## Final Model
Because the results of the Random Forest & Gradient Descent models were both extremely comprable in terms of RMSE, I'll focus on the Random Forest Model, because I feel that is more easily interpretable. 


-----

##Conclusion 