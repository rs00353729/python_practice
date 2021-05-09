# Last amended: 7th February, 2020
# objectives:
#           i)  Learning to draw various types of graphs
#          ii)  Conditional plots using catplot
#         iii)  Relationship plots using relplot
#          ii) Learning to use seaborn
# Good reference: https://seaborn.pydata.org/introduction.html

# 1.0 Call libraries
%reset -f
# 1.1 For data manipulations
import numpy as np
import pandas as pd
# 1.2 For plotting
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl     # For creating colormaps
import seaborn as sns
# 1.3 For data processing
from sklearn.preprocessing import StandardScaler as ss
# 1.4 OS related
import os

# 1.5 Go to folder containing data file
#os.chdir("D:\\data\\OneDrive\\Documents\\advertising")

os.chdir("D:\\Python\\Day5")
os.listdir()            # List all files in the folder

# 1.6 Read file and while reading file,
#      convert 'Timestamp' to datetime time
ad = pd.read_csv("advertising.zip",
                  parse_dates = ['Timestamp']    # especial for date parsing
                  )

# 1.7 Check data types of attributes
ad.dtypes
pd.set_option('display.max_columns', 500)
# 1.8 Some more dataset related information
ad.head(3)
ad.info()               # Also informs how much memory dataset takes
                        #   and status of nulls
ad.shape                # (1000, 10)
ad.columns.values
len(ad.columns)         # 10 attributes

# 1.9 Categorical data value counts
#     Or number of levels per category
len(ad.City.unique())                   # 969 cities out of 1000
ad.City.value_counts()

# 1.9.1 How many conutries
len(ad.Country.unique())                # 237 countries
ad.Country.value_counts()               # Mostly 2 per country

# 1.9.2 Distribution of gender
ad.Male.value_counts()                  # 519:481

# 1.9.3 Distribution of clicks
ad['Clicked on Ad'].value_counts()      # 1 and 0 in the ratio of 500:500
                                        # This is highly optimistic. Genrally clicks may be 1%

#############################
# 2.0 Create features
#############################
# 2.1 Descretise continuos columns
#     These are equal width bins as against
#     equal data-points bins (quantile) or kmeans clusters
#     Alternatively use KBinsDiscretizer of sklearn
ad["age_cat"] = pd.cut(
                       ad['Age'],
                       bins = 3,
                       labels= ["y", "m", "s"]
                      )

ad["area_income_cat"] = pd.cut(
                               ad['Area Income'],
                               bins = 3,
                               labels= ["l", "m", "h"]
                               )

# 2.2 Create a new column as per length of each ad-line
ad['AdTopicLineLength'] = ad['Ad Topic Line'].apply(lambda x : len(x))

# 2.3 Create a new column as per number of words in each ad-line
# Try "good boy".split(" ")  and len("good boy.split(" "))
"good boy".split(" ")             # ['good', 'boy']
len("good boy".split(" "))        # 2


# 2.3.1 Note the use of apply(). This apply() works on complete Series
#       to transform it rather than to summarise it as in groupby.
ad['AdTopicNoOfWords'] = ad['Ad Topic Line'].apply(lambda x : len(x.split(" ")))   # Note the use of apply()
                                                                                   # This apply works on complete Series


# 2.4 A column that has countd of City and
#       another column with count of Country columns
#       Note the use of transform method here
grouped = ad.groupby(['City'])
ad['City_count'] = grouped['City'].transform('count')   # count is a groupby method

# 2.4.1 Same way for country
grouped = ad.groupby(['Country'])
ad['Country_count'] = grouped['Country'].transform('count')   # count is a groupby method


# 2.5 Extract date components using Series.dt accessor
#     https://pandas.pydata.org/pandas-docs/stable/reference/series.html#api-series-dt
#     https://pandas.pydata.org/pandas-docs/stable/reference/series.html#datetime-properties

# 2.6 What is the type of 'dt'
type(ad['Timestamp'].dt)    # Accessor like get()
                            # pandas.core.indexes.accessors.DatetimeProperties

# 2.7 Extract hour, weekday and month
ad['hour']    = ad['Timestamp'].dt.hour
ad['weekday'] = ad['Timestamp'].dt.weekday
ad['quarter'] = ad['Timestamp'].dt.month      # First we get month. Then we map month to quarter
                                              #   See below

# 2.8 Transform hour to morning, evening, night etc
#     We use integers as levels
# Format: ad.loc[<condition>, featureName] = new_value
ad.loc[(ad['hour'] > 0 ) & (ad['hour'] <= 6),  'hour' ] = 0      # Early morning
ad.loc[(ad['hour'] > 6) & (ad['hour'] <= 12),  'hour' ] = 1      # Morning
ad.loc[(ad['hour'] >12 ) & (ad['hour'] <= 17), 'hour' ] = 2      # Afternoon
ad.loc[(ad['hour'] >17 ) & (ad['hour'] <= 20), 'hour' ] = 3     # Evening
ad.loc[(ad['hour'] >20 ) & (ad['hour'] <= 22), 'hour' ] = 4     # Night
ad.loc[(ad['hour'] >22 ) & (ad['hour'] <= 24), 'hour' ] = 5     # Late Night

ad['hour'].head()

# 2.9 Transform integer levels to their correct
#      respective names
#     Could have been done much easily using Series.map()
#     See below
ad['hour'] = ad['hour'].astype('object')
ad.dtypes.loc['hour']         # Check dtype of this column

ad.loc[ad['hour'] == 0, 'hour'] = "earlymorning"
ad.loc[ad['hour'] == 1, 'hour'] = "morning"
ad.loc[ad['hour'] == 2, 'hour'] = "afternoon"
ad.loc[ad['hour'] == 3, 'hour'] = "evening"
ad.loc[ad['hour'] == 4, 'hour'] = "night"
ad.loc[ad['hour'] == 4, 'hour'] = "night"
ad.loc[ad['hour'] == 5, 'hour'] = "latenight"

ad['timeOfDay'] = pd.cut(ad['hour'],bins=[-1,6,12,17,20,22,24],labels=['Early morning','morning','afternoon','evening','night','latenight'])

# 3.0 Similarly for weekdays
#     Map weekday numbers to weekday names
#     We use Series.map() method
ad['weekday'] = ad['weekday'].map({
                                    0 : 'Monday',
                                    1 : 'Tuesday',
                                    2: 'Wednesday',
                                    3: 'Thursday',
                                    4: 'Friday',
                                    5: 'Saturday',
                                    6: 'Sunday'
                                    }
                                )

ad['weekday'].head()

# 4.0 We use Series.map() method again but this time instead of supplying
#      a dictionary to dictate transformation, we use a function for
#        transformation
def month(x):
    if 0 < x < 3:
        return "Q1"            # Quarter 1
    if 3<= x < 6:
        return "Q2"            # Quarter 2
    if 6 <= x < 9:
        return "Q3"            # Quarter 3
    if 9 <= x < 12:
        return "Q4"            # Quarter 4

ad['quarter'] = ad['quarter'].map(lambda x : month(x))   # Which quarter clicked

#---- OR ------ 
ad['quarter'] = pd.cut(
ad['quarter'],
bins=[0,3,6,9,12],
labels=['Q1','Q2','Q3','Q4']
)

ad['quarter'].head()

# 4.1 So finally what are col names?
ad.columns.values
ad.shape               # (1000, 19)  Earlier shape was (1000, 10)

# 4.2 Let us rename some columns; remove spaces

new_col_names  = {
                 'Daily Time Spent on Site' :  'DailyTimeSpentonSite',
                 'Area Income'              : 'AreaIncome',
                 'Daily Internet Usage'     : 'DailyInternetUsage',
                 'Clicked on Ad'            : 'Clicked_on_ad',
                 'Male'                     : 'Gender'
              }
# 4.2.1
ad.rename(
         new_col_names,
         inplace = True,
         axis = 1             # Note the axis keyword. By default it is axis = 0
         )

ad.head()
ad.columns.values
# To print plot ina new quitified window
%matplotlib qt5 
# To print inline,
%matplotlib inline
##################
# 5 Plotting
##################

# Question 1: How is Age distributed?
# Question 2: How is DailyTimeSpentonSite distributed
# Question 3: How is AreaIncome distributed

# 5.1 Distribution of each continuous value using distplot()

# 5.1.1 Age is slight skewed to right. Naturally density of younger
#       persons is high
sns.distplot(ad.Age)

# 5.1.2 Add more plot configurations
# Refer: https://matplotlib.org/api/axes_api.html#matplotlib-axes
ax= sns.distplot(ad.Age)
ax.set( xlim =(10,80),
        xlabel= "age of persons",
        ylabel = "Denity",
        title= "Dencity of Age",
        xticks = list(range(0,80,5))
        )


# 5.1.2 Distribution of DailyTimeSpentonSite
sns.distplot(ad.DailyTimeSpentonSite)
sns.distplot(ad.AreaIncome)
sns.distplot(ad.DailyInternetUsage)

# Question 4: Show joint distribution of DailyTimeSpentonSite and AreaIncome
# Question 5: Show joint distribution of DailyInternetUsage and DailyTimeSpentonSite
# Question 6: Show these plots as kernel density as also 'hex' as also
#             draw regression line
#
# A jointplot = Scatterplot + Density plots

# 5.1.3 Open first the following
sns.jointplot(ad.DailyTimeSpentonSite, ad.AreaIncome)
# 5.1.4  and then this plot to understand meaning of colour intensity
#         in contour plots
sns.jointplot(ad.DailyTimeSpentonSite, ad.AreaIncome, kind = "kde")

# 5.1.5 Clearly two clusters are evident here
sns.jointplot(ad.DailyInternetUsage,
              ad.DailyTimeSpentonSite,
              kind = "kde"
              )

# 5.1.6 Or plot hex plot
sns.jointplot(ad.DailyInternetUsage,
              ad.DailyTimeSpentonSite,
              kind = "hex"
              )


# 5.1.7 Add regression and kernel density fits:
sns.jointplot(ad.DailyInternetUsage,
              ad.DailyTimeSpentonSite,
              kind = "reg"
              )


########################
# Discover Structure in data
# Question: 8 Does data have any pattern to predict 'Clicked_on_ad'
########################

# 6.0 Select only numeric columns for the purpose
num_columns = ad.select_dtypes(include = ['float64', 'int64']).copy()
num_columns.head()
num_columns.shape       # (1000, 10)

# 6.1 To this dataframe, add one more column of 'Clicked_on_ad'
#ad['Clicked_on_ad'] = ad['Clicked_on_ad'].astype('int8')
#num_columns['Clicked_on_ad'] = ad.loc[: , 'Clicked_on_ad']

# 6.1 Normalize  data
cols = ['DailyTimeSpentonSite', 'Age','AreaIncome', 'DailyInternetUsage', 'Gender', 'AdTopicLineLength', 'AdTopicNoOfWords', 'City_count', 'Country_count' ]

# 6.1.1 Create an instance of scaler object
ss= StandardScaler()
# 6.1.2 fit and transform
nc = ss.fit_transform(num_columns.loc[:,cols ])
# 6.1.3
nc.shape     # (1000,9)
# 6.1.4 Transform to dataframe
nc = pd.DataFrame(nc, columns = cols)
# 6.1.5 Add one more column (this column was not to be scaled)
nc['Clicked_on_ad'] = ad['Clicked_on_ad']

# 6.2 Next plot radviz, parallel_coordinates and andrews_curves
pd.plotting.radviz(nc,
                   class_column ='Clicked_on_ad',
                   colormap= 'winter'
                   )


pd.plotting.parallel_coordinates(nc,
                                 'Clicked_on_ad',
                                  colormap='winter'
                                  )

pd.plotting.andrews_curves(nc,
                           'Clicked_on_ad',
                           colormap = 'winter'
                           )


# 6.3 What if Data is random
rand = pd.DataFrame(np.random.randn(1000,9),
                    columns = cols    # Assign column names, just like that
                    )

# 6.3.1 Add this columns also
rand['Clicked_on_ad'] = ad['Clicked_on_ad']

# 6.3.2 Now start plotting
pd.plotting.parallel_coordinates(rand,
                                 'Clicked_on_ad',
                                  colormap='winter'
                                  )

pd.plotting.andrews_curves(rand,
                           'Clicked_on_ad',
                           colormap = 'winter')

##################
# Matrix plots or heatmap
##################

# Question 9:  Hour and weekday wise when are clicks most
# Question 10: Quarter wise and weekday wise when are clicks most
# Question 11: Quarter wise and weekday wise when are DailyInternetUsage max and min

# 7.0 When are total clicks more
#     Heatmap of hour vs weekday
#     X and Y labels are DataFrame indexes
grouped = ad.groupby(['hour', 'weekday'])
df_wh = grouped['Clicked_on_ad'].sum().unstack()
df_wh

# 7,1 Draw quickly the heatmap
sns.heatmap(df_wh)

# 7.1 Let us create a colourmap
#     Create first an appropriate colormap
#     See  https://stackoverflow.com/questions/52626103/custom-colormap
# Step 1. Determine limits of your data to create colormap
#          light to dark
df_wh.max().max()     # 30
df_wh.min().min()     # 0
# Step 2.
#  Normalize is a class which, when called, can normalize data into
#    the ``[0.0, 1.0]`` interval.
# 7.2 Create first an instance of this class
norm = matplotlib.colors.Normalize(0,30)
norm

# 7.3 Next define how color mapping will be done
mycolors = [
             [norm(0.0), "lightblue"],      # Minimum will be mapped to 'lightbue'
             [norm(30),  "darkblue" ]       # MAx will be mapped to 'darkblue'
           ]

# 7.4 Get color map now
cmap = mpl.colors.LinearSegmentedColormap.from_list("", mycolors)

# 7.5 Use this color map on 'hour vs weekday
sns.heatmap(df_wh, cmap = cmap)

# 7.6 Quarter vs weekday
grouped = ad.groupby(['weekday','quarter'])
df_wq = grouped['Clicked_on_ad'].sum().unstack()
sns.heatmap(df_wq, cmap = cmap)

# 7.7 In which quarter daily Internet usage is more
grouped = ad.groupby(['weekday', 'quarter'])
df_wqd = grouped['DailyInternetUsage'].mean().unstack()
df_wqd.max().max()          # 192
df_wqd.min().min()          # 169
norm = mpl.colors.Normalize(169,193)
mycolors = [[norm(169.0), "lightblue"],
            [norm(193.0), "darkblue"]]
cmap = mpl.colors.LinearSegmentedColormap.from_list("", mycolors)
sns.heatmap(df_wqd, cmap = cmap)

ad.columns.values


# Question 12: Age category wise, what is the distribution of AreaIncome

# 8.0 Show distribution of AreaIncome using boxplots
sns.boxplot(x = 'age_cat', y = 'AreaIncome', data = ad)

# Question 13: Age category wise, ad-clicked


# 8.1 Note how seaborn uses estimator function
#     Barplots are grouped summaries, category wise
#     'estimator' is a summary function
#       For errobars, see this wikpedia on bootstrap statistics
#         https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
sns.barplot(x = 'Gender',
            y = 'Clicked_on_ad',
            estimator = np.sum,      # As there are multiple occurrences of Gender, sum up 'Clicked_on_ad'
            data = ad
            )

# 8.2 Multiple ways of plotting similar information
sns.barplot(x = 'Gender',
            y = 'Clicked_on_ad',
            hue = 'age_cat',       # Age-cat wise plots
            estimator = np.sum,
            data =ad)

# 8.3 Facet plots
#     READ 'catplot' AS CONDITIONAL PLOTS

sns.catplot(x = 'Gender',
            y = 'Clicked_on_ad',
            hue = 'age_cat' ,
            row = 'area_income_cat',
            kind = 'bar',
            data = ad)

# 8.4
sns.catplot(x = 'age_cat',
            y = 'DailyInternetUsage',
            row = 'area_income_cat',
            col = 'Clicked_on_ad',
            estimator = np.mean ,
            kind = 'box',
            data =ad)

# Faceted scatter plots or relationship plots
sns.relplot(x = 'Age', y = 'DailyInternetUsage', row = 'area_income_cat', col = 'weekday', kind = 'scatter', data = ad)
sns.relplot(x = 'Age', y = 'DailyInternetUsage', hue = 'area_income_cat',  kind = 'scatter', data = ad, cmap = 'winter')
sns.relplot(x = 'Age', y = 'DailyInternetUsage', hue = 'area_income_cat', size = 'weekday', kind = 'scatter', data = ad)
sns.relplot(x = 'Age', y = 'DailyInternetUsage', hue = 'hour', kind = 'scatter', data = ad)
sns.relplot(x = 'Age', y = 'DailyInternetUsage', row = 'hour', kind = 'scatter', data = ad)

## AA. Plot density plots and boxplots to show which
##     attributes will be able to predict/classify
##     target attribute, Clicked_on_ad
## BB. Draw Boxen plots also
#      See Moodle under Machine Learning II to
#      know what is lvplot or boxen plots

sns.boxplot(x = 'Clicked_on_ad',y = 'DailyInternetUsage', data = ad)
 # For boxenplot, refer: https://chartio.com/learn/charts/box-plot-complete-guide/
sns.boxenplot(x = 'Clicked_on_ad', y = 'Age', data = ad)

# Draw conditional density plots
#  You have to draw overlapping plots as below
df = ad[ad['Clicked_on_ad'] == 0]
df1 = ad[ad['Clicked_on_ad'] == 1]
ax = sns.kdeplot(df.DailyInternetUsage, shade = True)
sns.kdeplot(df1.DailyInternetUsage, ax = ax, shade = True)
# Here is a conditional density plot
#  for a column with random values
ad['rand'] = np.random.randn(ad.shape[0])
df = ad[ad['Clicked_on_ad'] == 0]
df1 = ad[ad['Clicked_on_ad'] == 1]
ax = sns.kdeplot(df.rand, shade = True)
sns.kdeplot(df1.rand, ax = ax, shade = True)
sns.kdeplot(df1.rand, ax = ax, shade = True)
# And this is conditional boxplot for the 'rand' column
sns.boxplot(x = 'Clicked_on_ad', y = 'rand', data =ad)


####################
# A matplotlib colormap maps the numerical range between 0 and 1 to a range of colors.
# https://stackoverflow.com/a/47699278/3282777
import matplotlib.cm as cm
cm.register_cmap(name='mycmap',
                 data={'red':   [(0.,0,0),
                                 (1.,0,0)],

                       'green': [(0.,0.6,0.6),
                                 (1.,0.6,0.6)],

                       'blue':  [(0.,0.4,0.4),
                                 (1.,0.4,0.4)],

                       'alpha': [(0.,0,0),
                                 (1,1,1)]})

sns.heatmap(df_wqd, cmap = 'mycmap', vmin = 169, vmax = 193)
help(cm.register_cmap)

'alpha': [(0.,0,0),
          (0.2,0.4,0.5),
          (1,1,1)]})
