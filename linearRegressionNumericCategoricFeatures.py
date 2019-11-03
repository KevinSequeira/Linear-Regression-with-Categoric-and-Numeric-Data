## ==================================================================================== ##
## ==================================================================================== ##
## AUTHOR      : Kevin Sequeira                                                         ##
## DATE        : 31-October-2019                                                        ##
## FILE NAME   : Linear Regression from Scratch                                         ##
## LANGUAGE    : Python                                                                 ##
## VERSION     : 3.7.2                                                                  ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## In this program, we are going to fit a linear regression model to the "Melbourne     ##
## Housing Market" dataset, which you can find here:                                    ##
## https://www.kaggle.com/anthonypino/melbourne-housing-market/                         ##
##                                                                                      ##
## We will start by importing the dataset and analyzing the distribution of different   ##
## features. We will plot the features, run tests for correlation and association, and  ##
## identify the most important features that help estimate the value of a property. Af- ##
## ter that, we will fit a predictive model using Scikit-learn's LinearRegression mod-  ##
## ule. We will test the model fitted to our data over various seeds and examine the    ##
## fit statistics to determine the best model.                                          ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## Let's start with importing any necessary packages and functions we might require for ##
## the project.                                                                         ##
## ==================================================================================== ##
import warnings
warnings.filterwarnings("ignore")
import subprocess
subprocess.call(["pip", "install", "pandas"])
import pandas as pan
pan.set_option("display.max_columns", 10)
pan.set_option("display.width", 400)
print("Pandas version", pan.__version__, "imported")
subprocess.call(["pip", "install", "numpy"])
import numpy as np
print("NumPy version", np.__version__, "imported")
from datetime import datetime
import statistics as mystat
subprocess.call(["pip", "install", "seaborn"])
import seaborn as sb
subprocess.call(["pip", "install", "matplotlib"])
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy.stats import f_oneway, chi2, f
subprocess.call(["pip", "install", "sklearn"])
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
subprocess.call(["pip", "install", "imblearn"])
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## The next step is to import the dataset and store it in an appropriate format. Since  ##
## our dataset is originally a comma-separated values file, we can import it and store  ##
## int in a Pandas dataframe.                                                           ##
## ==================================================================================== ##
melbourneHousingData = pan.read_csv("MELBOURNE_HOUSE_PRICES_LESS.csv")
print("\n" + "Data set successfully imported.")
print("Number of rows:", melbourneHousingData.shape[0])
print("Number of columns:", melbourneHousingData.shape[1])
print("Columns names:", melbourneHousingData.columns.tolist())
## ==================================================================================== ##
## Output:                                                                              ##
## Data set successfully imported.                                                      ##
## Number of rows: 63023                                                                ##
## Number of columns: 13                                                                ##
## Columns names: ['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',  ##
##                 'Date', 'Postcode', 'Regionname', 'Propertycount', 'Distance',       ##
##                 'CouncilArea']                                                       ##
##                                                                                      ##
## And inspection of the column names will show you the target variable for "Price". We ##
## will be predicting the value of "Price" using linear regression. The remaining vari- ##
## ables will be our feature variables.                                                 ##
##                                                                                      ##
## Also, we might be able to use the column "Address" as a unique identifier.           ##
##                                                                                      ##
## The next step will be to define functions that we will use repeatedly over the cour- ##
## se of the project. Let's do that in the next section of code. We will keep coming    ##
## back to this section of code as the project progresses as we define new functions.   ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## In this section, we will define functions that we will use repeatedly over the cour- ##
## se of this project.                                                                  ##
## ==================================================================================== ##
def getDataFrameSummary(dataFrame):
    dataFrameColumns = pan.DataFrame()
    dataFrameColumns["Column Name"] = dataFrame.columns
    dataFrameColumns["Data Type"] = list(dataFrame.dtypes)
    dataFrameColumns["No. of Missing"] = list(dataFrame.isna().sum())
    dataFrameColumns["No. of Unique Values"] = [len(dataFrame[column].unique()) for column in dataFrame.columns]
    dataFrameColumns["No. of Duplicated Values"] = [sum(dataFrame[column].duplicated(keep = "first")) for column in dataFrame.columns]
    print("Number of Rows:", dataFrame.shape[0])
    print("Number of Rows:", dataFrame.shape[1])
    print(dataFrameColumns)
    del dataFrameColumns
    print()

def transformAndSortByDate(dataFrame):
    dataFrame["Postcode"] = dataFrame["Postcode"].astype(str)
    dataFrame["Date"] = dataFrame["Date"].apply(lambda row: datetime.strptime(row, '%d/%m/%Y'))
    dataFrame = dataFrame.sort_values("Date")
    return dataFrame

def dropDuplicates(dataFrame):
    dataFrame = dataFrame.drop_duplicates("Address", keep = "last")
    return dataFrame

def getDescriptiveStatistics(dataFrame, numericalColumns = [], categoricalColumns = [], datetimeColumns = []):
    numericalData = pan.DataFrame(columns = ["Feature", "Minimum", "Mean", "Median", "Mode", "Maximum"])
    categoricData = pan.DataFrame(columns = ["Feature", "Number of Labels", "Mode", "Mode Count"])
    datetimeData = pan.DataFrame()

    if len(numericalColumns) > 0:
        numericalData["Feature"] = numericalColumns
        for column in numericalColumns:
            numericalData["Minimum"][numericalData["Feature"] == column] = np.nanmin(dataFrame[column])
            numericalData["Mean"][numericalData["Feature"] == column] = np.nanmean(dataFrame[column])
            numericalData["Median"][numericalData["Feature"] == column] = np.nanmedian(dataFrame[column])
            numericalData["Mode"][numericalData["Feature"] == column] = mystat.mode(dataFrame[column])
            numericalData["Maximum"][numericalData["Feature"] == column] = format(np.nanmax(dataFrame[column]), ".0f")
        print("Descriptive Statistics for Numerical Features:")
        print(numericalData, "\n")

    if len(categoricalColumns) > 0:
        categoricData["Feature"] = categoricalColumns
        for column in categoricalColumns:
            categoricData["Number of Labels"][categoricData["Feature"] == column] = len(dataFrame[column].astype(str).unique())
            categoricData["Mode"][categoricData["Feature"] == column] = dataFrame[column].astype(str).value_counts().index[0]
            categoricData["Mode Count"][categoricData["Feature"] == column] = dataFrame[column].astype(str).value_counts()[0]
        print("Descriptive Statistics for Categorical Features:")
        print(categoricData, "\n")

def scatterPlot(xAxisData, yAxisData, xLabel, yLabel, plotTitle, figSize = (12, 6)):
    plt.figure(figsize = figSize)
    scatterPlot = sb.scatterplot(x = xAxisData, y = yAxisData)
    plt.xlabel("\n" + xLabel, fontsize = 13)
    plt.ylabel(yLabel + "\n", fontsize = 13)
    plt.xticks(rotation = 0)
    plt.ticklabel_format(style = "plain", axis = "y")
    plt.title("\n" + plotTitle + "\n", fontsize = 20)
    plt.show()

def kdePlot(target, dataColumn, xLabel, plotTitle):
    dataFrame = pan.DataFrame({xLabel: dataColumn, "Category": target}).dropna()
    plt.figure(figsize = (12, 6))
    for category in set(target):
        sb.kdeplot(dataFrame.loc[dataFrame["Category"] == category, xLabel], label = category)
    plt.xlabel("\n" + xLabel, fontsize = 13)
    plt.ylabel("Density" + "\n", fontsize = 13)
    plt.ticklabel_format(style = "plain", axis = "x")
    plt.title("\n" + plotTitle + "\n", fontsize = 20)
    plt.legend(title = "Property Type")
    plt.show()

def boxPlot(dataFrame, xAxisData, categories, xLabel, yLabel, plotTitle):
    plt.figure(figsize = (12, 6))
    boxPlot = sb.boxplot(x = categories, y = xAxisData, data = dataFrame)
    plt.xlabel("\n" + xLabel, fontsize = 13)
    plt.ylabel(yLabel + "\n", fontsize = 13)
    plt.xticks(rotation = 45)
    plt.ticklabel_format(style = "plain", axis = "y")
    plt.title("\n" + plotTitle + "\n", fontsize = 20)
    plt.show()

def visualizePriceByDistance(dataFrame):
    for houseType in dataFrame["Type"].unique():
        scatterPlot(xAxisData = dataFrame["Distance"][dataFrame["Type"] == houseType],
                    yAxisData = dataFrame["LogPrice"][dataFrame["Type"] == houseType],
                    xLabel = "Distance from Melbourne CBD",
                    yLabel = "Price of House in AU$",
                    plotTitle = "Price of Houses of Type '" + houseType + "' for varying Distance from Melbourne CBD")

def visualizePriceByPropertyCount(dataFrame):
    for houseType in dataFrame["Type"].unique():
        scatterPlot(xAxisData = dataFrame["Propertycount"][dataFrame["Type"] == houseType],
                    yAxisData = dataFrame["LogPrice"][dataFrame["Type"] == houseType],
                    xLabel = "Number of Properties in a Melbourne Suburb",
                    yLabel = "Price of House in AU$",
                    plotTitle = "Price of Houses of Type '" + houseType + "' by the number of properties in a Melbourne Suburb")

def visualizePriceByPropertyType(dataFrame):
    kdePlot(target = dataFrame["Type"],
            dataColumn = dataFrame["LogPrice"],
            xLabel = "Property Price in AU$",
            plotTitle = "Distribution of Property Price for different Property Types")

def visualizePriceByRooms(dataFrame):
    boxPlot(dataFrame = dataFrame,
            xAxisData = "LogPrice",
            categories = "Rooms",
            xLabel = "Property Price in AU$",
            yLabel = "No. of Rooms",
            plotTitle = "Distribution of Property Price by Number of Rooms")

def visualizePriceByMethod(dataFrame):
    boxPlot(dataFrame = dataFrame,
            xAxisData = "LogPrice",
            categories = "Method",
            xLabel = "Property Price in AU$",
            yLabel = "Method of Sale",
            plotTitle = "Distribution of Property Price by Method of Sale")

def visualizePriceByRegionName(dataFrame):
    boxPlot(dataFrame = dataFrame,
            xAxisData = "LogPrice",
            categories = "Regionname",
            xLabel = "Property Price in AU$",
            yLabel = "Region Name",
            plotTitle = "Distribution of Property Price by Region Name")

def visualizePriceByCouncilArea(dataFrame):
    boxPlot(dataFrame = dataFrame,
            xAxisData = "LogPrice",
            categories = "CouncilArea",
            xLabel = "Property Price in AU$",
            yLabel = "Council Area",
            plotTitle = "Distribution of Property Price by Council Area")

def visualizeNumericData(dataFrame):
    visualizePriceByDistance(dataFrame)
    visualizePriceByPropertyCount(dataFrame)

def visualizeCategoricData(dataFrame):
    visualizePriceByPropertyType(dataFrame)
    visualizePriceByRooms(dataFrame)
    visualizePriceByMethod(dataFrame)
    visualizePriceByRegionName(dataFrame)
    visualizePriceByCouncilArea(dataFrame)

def visualizeData(dataFrame):
    visualizeNumericData(dataFrame)
    visualizeCategoricData(dataFrame)

def logTransformPrice(dataFrame):
    dataFrame["LogPrice"] = dataFrame["Price"].apply(lambda value: np.log(value) if value != 0 else 0)
    dataFrame["LogRooms"] = dataFrame["Rooms"].apply(lambda value: np.log(value) if value != 0 else 0)
    dataFrame["LogDistance"] = dataFrame["Distance"].apply(lambda value: np.log(value) if value != 0 else 0)
    # dataFrame["RoomDistanceInteraction"] = dataFrame[["Rooms", "Distance"]].apply(lambda row: np.log(row[0] * row[1]) if (row[0] * row[1]) > 0 else 0, axis = 1)
    return dataFrame

def checkCorrelations(dataFrame, numericalFeatures, target):
    print("Correlation matrix between Numerical Features and Target:")
    print(dataFrame[numericalFeatures + [target]].corr())
    print("")

def checkAssociations(dataFrame, categoricFeatures, target):
    dataFrame = dataFrame.dropna()
    for feature in categoricFeatures:
        print("One-way ANOVA Test between", feature, "and", target + ":")
        parameterList = []
        dataFrame[feature] = dataFrame[feature].astype(str)
        for level in dataFrame[feature].unique():
            parameterList.append(dataFrame[target][dataFrame[feature] == level])
        fScore, pValue = f_oneway(*parameterList)
        print("F-score:", fScore)
        print("p-Value:", pValue)
    print("")

def mergeCategories(dataFrame, categoricFeatures, numberOfGroups, target):
    dataFrame = dataFrame.dropna()
    groupedFeatures = {}
    for feature, number in zip(categoricFeatures, numberOfGroups):
        groupedFeatures[feature] = {}
        labelEncoder = LabelEncoder()
        labelEncoder.fit(list(dataFrame[feature]))
        dataFrame["Grouped" + feature] = labelEncoder.transform(list(dataFrame[feature]))
        dataArray = np.array(dataFrame[["Grouped" + feature, target]])
        kmeans = KMeans(n_clusters = number, random_state = 0).fit(dataArray)
        dataFrame["Grouped" + feature] = kmeans.labels_
        dataFrame["Grouped" + feature] = dataFrame["Grouped" + feature].apply(lambda value: "Group " + str(value))
        for groupedFeature in dataFrame["Grouped" + feature].unique():
            groupedFeatures[feature][groupedFeature] = list(dataFrame[feature][dataFrame["Grouped" + feature] == groupedFeature].unique())
        groupedFeatureKeys = list(groupedFeatures[feature].keys())
        groupedFeatureKeysCopy = groupedFeatureKeys
        for group01 in groupedFeatureKeys:
            groupedFeatureKeysCopy.remove(group01)
            for group02 in groupedFeatureKeysCopy:
                commonElements = set(groupedFeatures[feature][group01]) & set(groupedFeatures[feature][group02])
                if len(commonElements) > 0:
                    for element in commonElements:
                        group01Count = len(dataFrame["Grouped" + feature][(dataFrame[feature] == element) & (dataFrame["Grouped" + feature] == group01)].index)
                        group02Count = len(dataFrame["Grouped" + feature][(dataFrame[feature] == element) & (dataFrame["Grouped" + feature] == group02)].index)
                        if group01Count >= group02Count:
                            groupedFeatures[feature][group02].remove(element)
                            dataFrame["Grouped" + feature][(dataFrame[feature] == element) & (dataFrame["Grouped" + feature] == group02)] = group01
                        else:
                            dataFrame["Grouped" + feature][(dataFrame[feature] == element) & (dataFrame["Grouped" + feature] == group01)] = group02
    return dataFrame, groupedFeatures

def visualizePriceByGroupedSuburb(dataFrame):
    boxPlot(dataFrame = dataFrame,
            xAxisData = "LogPrice",
            categories = "GroupedSuburb",
            xLabel = "Property Price in AU$",
            yLabel = "Suburb Group",
            plotTitle = "Distribution of Property Price by Suburb Group")

def visualizePriceBySellerGroup(dataFrame):
    boxPlot(dataFrame = dataFrame,
            xAxisData = "LogPrice",
            categories = "GroupedSellerG",
            xLabel = "Property Price in AU$",
            yLabel = "Seller Group",
            plotTitle = "Distribution of Property Price by Seller Group")

def visualizePriceByGroupedPostcode(dataFrame):
    boxPlot(dataFrame = dataFrame,
            xAxisData = "LogPrice",
            categories = "GroupedPostcode",
            xLabel = "Property Price in AU$",
            yLabel = "Seller Group",
            plotTitle = "Distribution of Property Price by Post Code Group")

def visualizePriceByGroupedCouncilArea(dataFrame):
    boxPlot(dataFrame = dataFrame,
            xAxisData = "LogPrice",
            categories = "GroupedCouncilArea",
            xLabel = "Property Price in AU$",
            yLabel = "Council Area Group",
            plotTitle = "Distribution of Property Price by Council Area Group")

def visualizeGroupedData(dataFrame):
    visualizePriceByGroupedSuburb(dataFrame)
    visualizePriceBySellerGroup(dataFrame)
    visualizePriceByGroupedPostcode(dataFrame)
    visualizePriceByGroupedCouncilArea(dataFrame)

def calculateRMSE(testTarget, predictedTarget):
    modelRMSE = np.sqrt(sum((testTarget - predictedTarget) ** 2) / len(testTarget))
    return modelRMSE

def modelRSquared(testTarget, predictedTarget):
    targetMean = np.mean(testTarget)
    totalSSE = np.sum((testTarget - targetMean) ** 2)
    residualSSE = np.sum((testTarget - predictedTarget) ** 2)
    rSquared = 1 - (residualSSE / totalSSE)
    return rSquared

def calculatePValueChiSquare(testTarget, predictedTarget):
    chiSquare = np.sum(((predictedTarget - testTarget) ** 2) / predictedTarget)
    pValueChiSquare = 1 - chi2.cdf(chiSquare, 1)
    print('Model Chi-Square: ', chiSquare)
    return pValueChiSquare

def calculatePValueFStatistic(testTarget, predictedTarget, degFreeRegression, degFreeResidual):
    meanRegressionSSE = np.sum((predictedTarget - np.mean(predictedTarget)) ** 2) / degFreeRegression
    meanResidualSSE = np.sum((testTarget - predictedTarget) ** 2) / degFreeResidual
    fStatistic = meanRegressionSSE / meanResidualSSE
    print('Model F-Statistic: ', fStatistic)
    pValueFStatistic = 1 - f.cdf(fStatistic, degFreeRegression, degFreeResidual)
    return pValueFStatistic

def fitAndTestModel(dataFrame, dataFeatures, dataCategoricalFeatures):
    dataPredictors = dataFrame[dataFeatures]
    dataPredictors = pan.get_dummies(dataPredictors, prefix = dataCategoricalFeatures)
    dataFeatures = dataPredictors.columns.tolist()
    dataPredictors = np.array(dataPredictors)
    print("Number of Features:", len(dataFeatures), "\n")
    dataTarget = np.array([melbourneHousingData["LogPrice"]]).transpose()
    # polynomialFeatues = PolynomialFeatures(degree = 2)
    # dataPredictors = polynomialFeatues.fit_transform(dataPredictors)
    for count in range(1, 11):
        print("SEED:", count)
        print("=====================================================================")
        dataPredictors_train, dataPredictors_test, dataTarget_train, dataTarget_test = train_test_split(StandardScaler().fit_transform(dataPredictors), dataTarget, test_size = 0.33, random_state = count)
        # dataPredictors_train, dataPredictors_test, dataTarget_train, dataTarget_test = train_test_split(dataPredictors, dataTarget, test_size = 0.33, random_state = count)

        linearRegression = LinearRegression()
        linearRegression.fit(dataPredictors_train, dataTarget_train)
        predictedTarget = linearRegression.predict(dataPredictors_test)
        print('Scikit Learn Model R-Squared: ', linearRegression.score(dataPredictors_test, dataTarget_test))
        modelRMSE = calculateRMSE(dataTarget_test, predictedTarget)
        print('Model RMSE: ', modelRMSE)
        # modelR2 = modelRSquared(dataTarget_test, predictedTarget)
        # print('Model R-Squared: ', modelR2)
        modelPValueChiSqTest = calculatePValueChiSquare(dataTarget_test, predictedTarget)
        print('Model P-Value for Chi-Square Test: ', modelPValueChiSqTest)
        modelPValueFStatistic = calculatePValueFStatistic(dataTarget_test, predictedTarget,
                                                          len(dataPredictors_train[0]),
                                                          len(dataPredictors_train) - len(dataPredictors_train[0]) - 1)
        print('Model P-Value for F-Statistic: ', modelPValueFStatistic)
        print()
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## Step 1: Let's get a summary of the dataframe. This will give us the data type of a   ##
## column, number of rows, number of columns, number of missing values, number of uni-  ##
## que values, and number of duplicated values.                                         ##
## ==================================================================================== ##
getDataFrameSummary(melbourneHousingData)
## ==================================================================================== ##
## Output:                                                                              ##
## Number of Rows: 63023                                                                ##
## Number of Columns: 13                                                                ##
##       Column Name Data Type  ...  No. of Unique Values  No. of Duplicated Values     ##
## 0          Suburb    object  ...                   380                     62643     ##
## 1         Address    object  ...                 57754                      5269     ##
## 2           Rooms     int64  ...                    14                     63009     ##
## 3            Type    object  ...                     3                     63020     ##
## 4           Price   float64  ...                  3418                     59605     ##
## 5          Method    object  ...                     9                     63014     ##
## 6         SellerG    object  ...                   476                     62547     ##
## 7            Date    object  ...                   112                     62911     ##
## 8        Postcode     int64  ...                   225                     62798     ##
## 9      Regionname    object  ...                     8                     63015     ##
## 10  Propertycount     int64  ...                   368                     62655     ##
## 11       Distance   float64  ...                   180                     62843     ##
## 12    CouncilArea    object  ...                    34                     62989     ##
##                                                                                      ##
## The dataframe summary table shown above indicates that 5269 addresses have been rep- ##
## eated. In the next step, we'll identify and remove the duplicated addresses keeping  ##
## only the last entry among each set of duplicates. Before that, we will sort the data ##
## by "Date" it was entered.                                                            ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## Step 2: Sort the data by "Date". Next, identify and remove the duplicated addresses  ##
## keeping only the last entry among each set of duplicates. After dropping duplicates  ##
## get the dataframe summary like before.                                               ##
## ==================================================================================== ##
melbourneHousingData = melbourneHousingData.dropna()
melbourneHousingData = transformAndSortByDate(melbourneHousingData)
melbourneHousingData = dropDuplicates(melbourneHousingData)
getDataFrameSummary(melbourneHousingData)
## ==================================================================================== ##
## Output:                                                                              ##
## Number of Rows: 57754                                                                ##
## Number of Rows: 13                                                                   ##
##       Column Name  ... No. of Duplicated Values                                      ##
## 0          Suburb  ...                    57374                                      ##
## 1         Address  ...                        0                                      ##
## 2           Rooms  ...                    57740                                      ##
## 3            Type  ...                    57751                                      ##
## 4           Price  ...                    54373                                      ##
## 5          Method  ...                    57745                                      ##
## 6         SellerG  ...                    57284                                      ##
## 7            Date  ...                    57646                                      ##
## 8        Postcode  ...                    57529                                      ##
## 9      Regionname  ...                    57746                                      ##
## 10  Propertycount  ...                    57386                                      ##
## 11       Distance  ...                    57574                                      ##
## 12    CouncilArea  ...                    57720                                      ##
##                                                                                      ##
## The latest dataframe summary shows that rows with duplicated "Address" values have   ##
## been dropped from the dataset. Consequently, the total number of rows in the dataset ##
## have gone down by the equivalent number.                                             ##
##                                                                                      ##
## We can now go forward and start analyzing the data in our dataset. It is good to     ##
## begin with looking at descriptive statistics. Descriptive statistics gives us info-  ##
## mation such as averages and modes in a dataset, as well as minimum and maximum of a  ##
## feature.                                                                             ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## Step 3: Analyze the data with descriptive statistics. Descriptive statistics give us ##
## infomation such as averages and modes in a dataset, as well as minimum and maximum   ##
## of a feature.                                                                        ##
## ==================================================================================== ##
getDescriptiveStatistics(melbourneHousingData,
                         numericalColumns = ["Price", "Propertycount", "Distance"],
                         categoricalColumns = ["Suburb", "Rooms", "Type", "Method", "SellerG", "Postcode", "Regionname", "CouncilArea"],
                         datetimeColumns = ["Date"])
## ==================================================================================== ##
## Descriptive Statistics for Numerical Features:                                       ##
##          Feature Minimum     Mean  Median    Mode   Maximum                          ##
## 0          Price   85000   994799  830000  600000  11200000                          ##
## 1  Propertycount      39  7599.57    6786   21650     21650                          ##
## 2       Distance       0  12.6705    11.7    10.5        64                          ##
##                                                                                      ##
## The table "Descriptive Statistics for Numerical Features" shows the range of values  ##
## for the features "Price", "Propertycount", and "Distance". We see the difference in  ##
## scale for each
##                                                                                      ##
## Descriptive Statistics for Categorical Features:                                     ##
##          Feature Number of Labels                     Mode Mode Count                ##
## 0         Suburb              380                Reservoir       1127                ##
## 1          Rooms               14                        3      25535                ##
## 2           Type                3                        h      41079                ##
## 3         Method                9                        S      31489                ##
## 4        SellerG              470                    Barry       6225                ##
## 5       Postcode              225                     3073       1127                ##
## 6     Regionname                8    Southern Metropolitan      16214                ##
## 7    CouncilArea               34  Boroondara City Council       4716                ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## It is good to start with looking at scatterplots and boxplots that give you an idea  ##
## of how your feature values are distributed with regards to the target. We can plot   ##
## the price of a house against features that are numeric using scatter plots, while    ##
## categorical features can be analyzed using box plots or a density plot.              ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## Step 4: We need to visualize how the data is distributed. We'll create a super-func- ##
## tion called "visualizeData( )", and two inner-functions for visualizing numeric data ##
## and categoric data.                                                                  ##
## ==================================================================================== ##
# visualizeData(melbourneHousingData)
## ==================================================================================== ##
## Both, the descriptive statistics and the scatter plots for numeric features show the ##
## discrepancy in the range of values for each feature. We see that property price can  ##
## take values between AU$85,000 to above AU$10,00,000, as compared to property dist-   ##
## ance from Melbourne CBD and number of rooms which take values between 0 to 70 and 1  ##
## to 31 respectively. The density plot for property price by property type shows that  ##
## distribution of price is mostly left skewed. The same goes for property distance     ##
## from Melbourne CBD and number of rooms. In such a case, it helps to take a log tran- ##
## sform of respective numeric features.                                                ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## Step 5: Transform the data. Apply log transformation to "Price" and / or "Distance"  ##
## and "Rooms" in order to remove any skewness in the data. Visualize the distributions ##
## again to check if there have been any improvements.                                  ##
## ==================================================================================== ##
melbourneHousingData = logTransformPrice(melbourneHousingData)
# visualizeData(melbourneHousingData)
## ==================================================================================== ##
## The distributions do seem slightly improved. The scatterplots now seem less disper-  ##
## sed and the density plots have improved in terms of skewness. The next step would be ##
## to look at the correlation matrix for numeric features and the target variable and   ##
## check for association between categoric features and the target variable.            ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## Step 6: Check for association and correlation between features and the target vari-  ##
## able "Price". For checking correlations, we use the ".corr( )" function in Pandas.   ##
## For checking association, we use the ".f_oneway( )" function from the package SciPy  ##
## Stats. We create two functions, "checkCorrelations( )" and "checkAssociations( )" in ##
## which we carry out necessary operations.                                             ##
## ==================================================================================== ##
checkCorrelations(melbourneHousingData,
                  numericalFeatures = ["LogRooms", "Propertycount", "LogDistance"],
                  target = "LogPrice")
checkAssociations(melbourneHousingData,
                  categoricFeatures = ["Suburb", "Type",
                                       "Method", "SellerG", "Postcode",
                                       "Regionname", "CouncilArea"],
                  target = "LogPrice")
## ==================================================================================== ##
## The correlation matrix for numeric features shows that all features are weakly-to-   ##
## moderately correlated to each other. The strongest correlation is between the target ##
## "LogPrice" and "LogRooms" at 0.480135. This shows that as the number of rooms in a   ##
## property increase, the property price is also expected to increase. Also, we see a   ##
## weak negative correlation between "LogDistance" and "LogPrice", indicating that as   ##
## the property distance from Melbourne CBD increases, the property price decreases but ##
## not drastically.                                                                     ##
##                                                                                      ##
## Correlation matrix between Numerical Features and Target:                            ##
##                LogRooms  Propertycount  LogDistance  LogPrice                        ##
## LogRooms       1.000000      -0.065993     0.372125  0.480135                        ##
## Propertycount -0.065993       1.000000    -0.039730 -0.085819                        ##
## LogDistance    0.372125      -0.039730     1.000000 -0.204360                        ##
## LogPrice       0.480135      -0.085819    -0.204360  1.000000                        ##
##                                                                                      ##
## The output for one-way ANOVA test of association for categorical features shows how  ##
## the categoric features influence the target "LogPrice". The F-score returned for the ##
## test indicates level of association. From the results shown below, we see that the   ##
## feature "Type" is highly associated with "LogPrice", followed by "RegionName", "Cou- ##
## ncilArea" and "Method". In relation to these, the features "Suburb", "SellerG" and   ##
## "Postcode" have really weak association.                                             ##
##                                                                                      ##
## One-way ANOVA Test between Suburb and LogPrice:                                      ##
## F-score: 78.85385339974466                                                           ##
## p-Value: 0.0                                                                         ##
## One-way ANOVA Test between Type and LogPrice:                                        ##
## F-score: 4442.383743808431                                                           ##
## p-Value: 0.0                                                                         ##
## One-way ANOVA Test between Method and LogPrice:                                      ##
## F-score: 229.25099162608842                                                          ##
## p-Value: 3.483995532577828e-195                                                      ##
## One-way ANOVA Test between SellerG and LogPrice:                                     ##
## F-score: 33.799744521361276                                                          ##
## p-Value: 0.0                                                                         ##
## One-way ANOVA Test between Postcode and LogPrice:                                    ##
## F-score: 123.98845748287972                                                          ##
## p-Value: 0.0                                                                         ##
## One-way ANOVA Test between Regionname and LogPrice:                                  ##
## F-score: 1326.704992343778                                                           ##
## p-Value: 0.0                                                                         ##
## One-way ANOVA Test between CouncilArea and LogPrice:                                 ##
## F-score: 545.5590947352435                                                           ##
## p-Value: 0.0                                                                         ##
##                                                                                      ##
## One of the reasons for weak associations between the features "Postcode", "Council-  ##
## Area", "SellerG", "Suburb" and the targer "LogPrice" is the large number of categor- ##
## ies available in each feature. It is possible to improve association by merging some ##
## some of the categories within each categorical feature. I have written a basic func- ##
## tion that can reduce the number of levels in each categorical feature down to a num- ##
## that you prefer.                                                                     ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## Step 7: Reduce number of categories in the weakly associated features. After the     ##
## categories are reduced, check for associations again and note if there is an improv- ##
## ement in results returned for the one-way ANOVA tests.                               ##
## ==================================================================================== ##
melbourneHousingData, groupedFeaturesDict = mergeCategories(melbourneHousingData,
                                                            categoricFeatures = ["Suburb", "SellerG", "Postcode", "CouncilArea"],
                                                            numberOfGroups = [10, 10, 10, 15],
                                                            target = "LogPrice")
checkAssociations(melbourneHousingData,
                  categoricFeatures = ["GroupedSuburb", "Type",
                                       "Method", "GroupedSellerG", "GroupedPostcode",
                                       "Regionname", "GroupedCouncilArea"],
                  target = "LogPrice")
# visualizeGroupedData(melbourneHousingData)
## ==================================================================================== ##
## One-way ANOVA Test between GroupedSuburb and LogPrice:                               ##
## F-score: 191.67456589237952                                                          ##
## p-Value: 0.0                                                                         ##
## One-way ANOVA Test between Type and LogPrice:                                        ##
## F-score: 4442.383743808431                                                           ##
## p-Value: 0.0                                                                         ##
## One-way ANOVA Test between Method and LogPrice:                                      ##
## F-score: 229.25099162608842                                                          ##
## p-Value: 3.483995532577828e-195                                                      ##
## One-way ANOVA Test between GroupedSellerG and LogPrice:                              ##
## F-score: 386.9707092688109                                                           ##
## p-Value: 0.0                                                                         ##
## One-way ANOVA Test between GroupedPostcode and LogPrice:                             ##
## F-score: 809.7616164655165                                                           ##
## p-Value: 0.0                                                                         ##
## One-way ANOVA Test between Regionname and LogPrice:                                  ##
## F-score: 1326.704992343778                                                           ##
## p-Value: 0.0                                                                         ##
## One-way ANOVA Test between GroupedCouncilArea and LogPrice:                          ##
## F-score: 938.7659103454372                                                           ##
## p-Value: 0.0                                                                         ##
##                                                                                      ##
## It is evident that after the categories in the variables "Suburb", "SellerG", "Post- ##
## code" and "CouncilArea" were grouped to give lesser number of categories, the one-   ##
## way ANOVA returned better F-scores for association tests. The p-values for these     ##
## tests are all well below the 0.05 significance level, which shows that these results ##
## are not obtained by chance, and are valid.                                           ##
##                                                                                      ##
## We can now go forward and fit a linear regression model the data.                    ##
## ==================================================================================== ##
## ==================================================================================== ##

## ==================================================================================== ##
## ==================================================================================== ##
## Step 8: Fit a linear model to the data. Before we fit a model, we one-hot encode the ##
## categorical features to give us sparse columns that are set to "1" if the category   ##
## exists for the row in the original data feature, and "0" otherwise. The data is then ##
## split into train and test datasets at a 2:1 ratio, and normalized before a linear    ##
## model is attached to it using Scikit-learn's LinearRegression module. This process   ##
## is repeated over 10 different SEED values, and the model fit statistics are compared ##
## to guage how well a linear regression model can explain variability in the target    ##
## "LogPrice". The entire process is coded inside the function "fitAndTestModel( )".    ##
## ==================================================================================== ##
## dataFeatures = ["GroupedSuburb", "Type", "Method", "GroupedSellerG", "GroupedPostcode", "Regionname", "GroupedCouncilArea", "Distance", "Rooms", "Propertycount"]
dataFeatures = ["LogDistance", "LogRooms", "Type", "Regionname", "GroupedCouncilArea", "GroupedPostcode", "GroupedSellerG", "Method", "GroupedSuburb"]
## dataCategoricalFeatures = ["GroupedSuburb", "Type", "Method", "GroupedSellerG", "GroupedPostcode", "Regionname", "GroupedCouncilArea"]
dataCategoricalFeatures = ["Type", "Regionname", "GroupedCouncilArea", "GroupedPostcode", "GroupedSellerG", "Method", "GroupedSuburb"]
fitAndTestModel(melbourneHousingData, dataFeatures, dataCategoricalFeatures)
## ==================================================================================== ##
## Number of Features: 63                                                               ##
##                                                                                      ##
## SEED: 1                                                                              ##
## =====================================================================                ##
## Scikit Learn Model R-Squared:  0.7140086566369328                                    ##
## Model RMSE:  [0.26592324]                                                            ##
## Model Chi-Square:  76.1078163338649                                                  ##
## Model P-Value for Chi-Square Test:  0.0                                              ##
## Model F-Statistic:  1191.4203843971234                                               ##
## Model P-Value for F-Statistic:  1.1102230246251565e-16                               ##
##                                                                                      ##
## SEED: 2                                                                              ##
## =====================================================================                ##
## Scikit Learn Model R-Squared:  0.7222749780722648                                    ##
## Model RMSE:  [0.26321263]                                                            ##
## Model Chi-Square:  74.59333406827771                                                 ##
## Model P-Value for Chi-Square Test:  0.0                                              ##
## Model F-Statistic:  1241.4551790143164                                               ##
## Model P-Value for F-Statistic:  1.1102230246251565e-16                               ##
##                                                                                      ##
## SEED: 3                                                                              ##
## =====================================================================                ##
## Scikit Learn Model R-Squared:  0.7188981894438486                                    ##
## Model RMSE:  [0.26483361]                                                            ##
## Model Chi-Square:  75.46434222121597                                                 ##
## Model P-Value for Chi-Square Test:  0.0                                              ##
## Model F-Statistic:  1215.143297180785                                                ##
## Model P-Value for F-Statistic:  1.1102230246251565e-16                               ##
##                                                                                      ##
## SEED: 4                                                                              ##
## =====================================================================                ##
## Scikit Learn Model R-Squared:  0.7179756045565394                                    ##
## Model RMSE:  [0.26420154]                                                            ##
## Model Chi-Square:  75.17942908013394                                                 ##
## Model P-Value for Chi-Square Test:  0.0                                              ##
## Model F-Statistic:  1226.1578263978138                                               ##
## Model P-Value for F-Statistic:  1.1102230246251565e-16                               ##
##                                                                                      ##
## SEED: 5                                                                              ##
## =====================================================================                ##
## Scikit Learn Model R-Squared:  0.7213315996786414                                    ##
## Model RMSE:  [0.26398319]                                                            ##
## Model Chi-Square:  74.89278996209333                                                 ##
## Model P-Value for Chi-Square Test:  0.0                                              ##
## Model F-Statistic:  1222.4444567236385                                               ##
## Model P-Value for F-Statistic:  1.1102230246251565e-16                               ##
##                                                                                      ##
## SEED: 6                                                                              ##
## =====================================================================                ##
## Scikit Learn Model R-Squared:  0.7103970258302581                                    ##
## Model RMSE:  [0.26686486]                                                            ##
## Model Chi-Square:  76.64983750532897                                                 ##
## Model P-Value for Chi-Square Test:  0.0                                              ##
## Model F-Statistic:  1195.2072751928674                                               ##
## Model P-Value for F-Statistic:  1.1102230246251565e-16                               ##
##                                                                                      ##
## SEED: 7                                                                              ##
## =====================================================================                ##
## Scikit Learn Model R-Squared:  0.7208787907736486                                    ##
## Model RMSE:  [0.2651212]                                                             ##
## Model Chi-Square:  75.6252194327353                                                  ##
## Model P-Value for Chi-Square Test:  0.0                                              ##
## Model F-Statistic:  1206.19557025307                                                 ##
## Model P-Value for F-Statistic:  1.1102230246251565e-16                               ##
##                                                                                      ##
## SEED: 8                                                                              ##
## =====================================================================                ##
## Scikit Learn Model R-Squared:  0.7183945483156646                                    ##
## Model RMSE:  [0.26460413]                                                            ##
## Model Chi-Square:  75.28704828967442                                                 ##
## Model P-Value for Chi-Square Test:  0.0                                              ##
## Model F-Statistic:  1189.5185908655224                                               ##
## Model P-Value for F-Statistic:  1.1102230246251565e-16                               ##
##                                                                                      ##
## SEED: 9                                                                              ##
## =====================================================================                ##
## Scikit Learn Model R-Squared:  0.7122594179258827                                    ##
## Model RMSE:  [0.26702701]                                                            ##
## Model Chi-Square:  76.67268806154985                                                 ##
## Model P-Value for Chi-Square Test:  0.0                                              ##
## Model F-Statistic:  1183.7064884295592                                               ##
## Model P-Value for F-Statistic:  1.1102230246251565e-16                               ##
##                                                                                      ##
## SEED: 10                                                                             ##
## =====================================================================                ##
## Scikit Learn Model R-Squared:  0.7181600764535334                                    ##
## Model RMSE:  [0.26579991]                                                            ##
## Model Chi-Square:  75.96760182309269                                                 ##
## Model P-Value for Chi-Square Test:  0.0                                              ##
## Model F-Statistic:  1193.747617936847                                                ##
## Model P-Value for F-Statistic:  1.1102230246251565e-16                               ##
##                                                                                      ##
## ==================================================================================== ##
## ==================================================================================== ##
