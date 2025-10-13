### Required Assignment 5.1: Will the Customer Accept the Coupon?

**Context**

Imagine driving through town and a coupon is delivered to your cell phone for a restaurant near where you are driving. Would you accept that coupon and take a short detour to the restaurant? Would you accept the coupon but use it on a subsequent trip? Would you ignore the coupon entirely? What if the coupon was for a bar instead of a restaurant? What about a coffee house? Would you accept a bar coupon with a minor passenger in the car? What about if it was just you and your partner in the car? Would weather impact the rate of acceptance? What about the time of day?

Obviously, proximity to the business is a factor on whether the coupon is delivered to the driver or not, but what are the factors that determine whether a driver accepts the coupon once it is delivered to them? How would you determine whether a driver is likely to accept a coupon?

**Overview**

The goal of this project is to use what you know about visualizations and probability distributions to distinguish between customers who accepted a driving coupon versus those that did not.

**Data**

This data comes to us from the UCI Machine Learning repository and was collected via a survey on Amazon Mechanical Turk. The survey describes different driving scenarios including the destination, current time, weather, passenger, etc., and then ask the person whether he will accept the coupon if he is the driver. Answers that the user will drive there ‘right away’ or ‘later before the coupon expires’ are labeled as ‘Y = 1’ and answers ‘no, I do not want the coupon’ are labeled as ‘Y = 0’.  There are five different types of coupons -- less expensive restaurants (under \$20), coffee houses, carry out & take away, bar, and more expensive restaurants (\$20 - $50).

**Deliverables**

Your final product should be a brief report that highlights the differences between customers who did and did not accept the coupons.  To explore the data you will utilize your knowledge of plotting, statistical summaries, and visualization using Python. You will publish your findings in a public facing github repository as your first portfolio piece.





### Data Description
Keep in mind that these values mentioned below are average values.

The attributes of this data set include:
1. User attributes
    -  Gender: male, female
    -  Age: below 21, 21 to 25, 26 to 30, etc.
    -  Marital Status: single, married partner, unmarried partner, or widowed
    -  Number of children: 0, 1, or more than 1
    -  Education: high school, bachelors degree, associates degree, or graduate degree
    -  Occupation: architecture & engineering, business & financial, etc.
    -  Annual income: less than \\$12500, \\$12500 - \\$24999, \\$25000 - \\$37499, etc.
    -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    -  Number of times that he/she buys takeaway food: 0, less than 1, 1 to 3, 4 to 8 or greater
    than 8
    -  Number of times that he/she goes to a coffee house: 0, less than 1, 1 to 3, 4 to 8 or
    greater than 8
    -  Number of times that he/she eats at a restaurant with average expense less than \\$20 per
    person: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    -  Number of times that he/she goes to a bar: 0, less than 1, 1 to 3, 4 to 8 or greater than 8
    

2. Contextual attributes
    - Driving destination: home, work, or no urgent destination
    - Location of user, coupon and destination: we provide a map to show the geographical
    location of the user, destination, and the venue, and we mark the distance between each
    two places with time of driving. The user can see whether the venue is in the same
    direction as the destination.
    - Weather: sunny, rainy, or snowy
    - Temperature: 30F, 55F, or 80F
    - Time: 10AM, 2PM, or 6PM
    - Passenger: alone, partner, kid(s), or friend(s)


3. Coupon attributes
    - time before it expires: 2 hours or one day


```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
```

### Problems

Use the prompts below to get started with your data analysis.  

1. Read in the `coupons.csv` file.





```python
data = pd.read_csv('data/coupons.csv')
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>destination</th>
      <th>passanger</th>
      <th>weather</th>
      <th>temperature</th>
      <th>time</th>
      <th>coupon</th>
      <th>expiration</th>
      <th>gender</th>
      <th>age</th>
      <th>maritalStatus</th>
      <th>...</th>
      <th>CoffeeHouse</th>
      <th>CarryAway</th>
      <th>RestaurantLessThan20</th>
      <th>Restaurant20To50</th>
      <th>toCoupon_GEQ5min</th>
      <th>toCoupon_GEQ15min</th>
      <th>toCoupon_GEQ25min</th>
      <th>direction_same</th>
      <th>direction_opp</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No Urgent Place</td>
      <td>Alone</td>
      <td>Sunny</td>
      <td>55</td>
      <td>2PM</td>
      <td>Restaurant(&lt;20)</td>
      <td>1d</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Coffee House</td>
      <td>2h</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Carry out &amp; Take away</td>
      <td>2h</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>2PM</td>
      <td>Coffee House</td>
      <td>2h</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>2PM</td>
      <td>Coffee House</td>
      <td>1d</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12679</th>
      <td>Home</td>
      <td>Partner</td>
      <td>Rainy</td>
      <td>55</td>
      <td>6PM</td>
      <td>Carry out &amp; Take away</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12680</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Rainy</td>
      <td>55</td>
      <td>7AM</td>
      <td>Carry out &amp; Take away</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12681</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Snowy</td>
      <td>30</td>
      <td>7AM</td>
      <td>Coffee House</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12682</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Snowy</td>
      <td>30</td>
      <td>7AM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12683</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Sunny</td>
      <td>80</td>
      <td>7AM</td>
      <td>Restaurant(20-50)</td>
      <td>2h</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12684 rows × 26 columns</p>
</div>



2. Investigate the dataset for missing or problematic data.


```python
# check for each column the value counts so to understand how the data is and if there is any issue.
for dest in data.columns:
    print(data[dest].value_counts())
```

    destination
    No Urgent Place    6283
    Home               3237
    Work               3164
    Name: count, dtype: int64
    passanger
    Alone        7305
    Friend(s)    3298
    Partner      1075
    Kid(s)       1006
    Name: count, dtype: int64
    weather
    Sunny    10069
    Snowy     1405
    Rainy     1210
    Name: count, dtype: int64
    temperature
    80    6528
    55    3840
    30    2316
    Name: count, dtype: int64
    time
    6PM     3230
    7AM     3164
    10AM    2275
    2PM     2009
    10PM    2006
    Name: count, dtype: int64
    coupon
    Coffee House             3996
    Restaurant(<20)          2786
    Carry out & Take away    2393
    Bar                      2017
    Restaurant(20-50)        1492
    Name: count, dtype: int64
    expiration
    1d    7091
    2h    5593
    Name: count, dtype: int64
    gender
    Female    6511
    Male      6173
    Name: count, dtype: int64
    age
    21         2653
    26         2559
    31         2039
    50plus     1788
    36         1319
    41         1093
    46          686
    below21     547
    Name: count, dtype: int64
    maritalStatus
    Married partner      5100
    Single               4752
    Unmarried partner    2186
    Divorced              516
    Widowed               130
    Name: count, dtype: int64
    has_children
    0    7431
    1    5253
    Name: count, dtype: int64
    education
    Some college - no degree                  4351
    Bachelors degree                          4335
    Graduate degree (Masters or Doctorate)    1852
    Associates degree                         1153
    High School Graduate                       905
    Some High School                            88
    Name: count, dtype: int64
    occupation
    Unemployed                                   1870
    Student                                      1584
    Computer & Mathematical                      1408
    Sales & Related                              1093
    Education&Training&Library                    943
    Management                                    838
    Office & Administrative Support               639
    Arts Design Entertainment Sports & Media      629
    Business & Financial                          544
    Retired                                       495
    Food Preparation & Serving Related            298
    Healthcare Practitioners & Technical          244
    Healthcare Support                            242
    Community & Social Services                   241
    Legal                                         219
    Transportation & Material Moving              218
    Architecture & Engineering                    175
    Personal Care & Service                       175
    Protective Service                            175
    Life Physical Social Science                  170
    Construction & Extraction                     154
    Installation Maintenance & Repair             133
    Production Occupations                        110
    Building & Grounds Cleaning & Maintenance      44
    Farming Fishing & Forestry                     43
    Name: count, dtype: int64
    income
    $25000 - $37499     2013
    $12500 - $24999     1831
    $37500 - $49999     1805
    $100000 or More     1736
    $50000 - $62499     1659
    Less than $12500    1042
    $87500 - $99999      895
    $75000 - $87499      857
    $62500 - $74999      846
    Name: count, dtype: int64
    car
    Scooter and motorcycle                      22
    Mazda5                                      22
    do not drive                                22
    crossover                                   21
    Car that is too old to install Onstar :D    21
    Name: count, dtype: int64
    Bar
    never    5197
    less1    3482
    1~3      2473
    4~8      1076
    gt8       349
    Name: count, dtype: int64
    CoffeeHouse
    less1    3385
    1~3      3225
    never    2962
    4~8      1784
    gt8      1111
    Name: count, dtype: int64
    CarryAway
    1~3      4672
    4~8      4258
    less1    1856
    gt8      1594
    never     153
    Name: count, dtype: int64
    RestaurantLessThan20
    1~3      5376
    4~8      3580
    less1    2093
    gt8      1285
    never     220
    Name: count, dtype: int64
    Restaurant20To50
    less1    6077
    1~3      3290
    never    2136
    4~8       728
    gt8       264
    Name: count, dtype: int64
    toCoupon_GEQ5min
    1    12684
    Name: count, dtype: int64
    toCoupon_GEQ15min
    1    7122
    0    5562
    Name: count, dtype: int64
    toCoupon_GEQ25min
    0    11173
    1     1511
    Name: count, dtype: int64
    direction_same
    0    9960
    1    2724
    Name: count, dtype: int64
    direction_opp
    1    9960
    0    2724
    Name: count, dtype: int64
    Y
    1    7210
    0    5474
    Name: count, dtype: int64


3. Decide what to do about your missing data -- drop, replace, other...


```python
# rename passanger column to passenger
data.rename(columns ={"passanger" : "passenger"})
data.loc[:, 'age'] = data['age'].str.replace("below21", "20")
data.loc[:, 'age'] = data['age'].str.replace("50plus", "20")
data.loc[:, 'age'] = data['age'].astype(int)
```


```python
# drop the duplicates
data.drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>destination</th>
      <th>passanger</th>
      <th>weather</th>
      <th>temperature</th>
      <th>time</th>
      <th>coupon</th>
      <th>expiration</th>
      <th>gender</th>
      <th>age</th>
      <th>maritalStatus</th>
      <th>...</th>
      <th>CoffeeHouse</th>
      <th>CarryAway</th>
      <th>RestaurantLessThan20</th>
      <th>Restaurant20To50</th>
      <th>toCoupon_GEQ5min</th>
      <th>toCoupon_GEQ15min</th>
      <th>toCoupon_GEQ25min</th>
      <th>direction_same</th>
      <th>direction_opp</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No Urgent Place</td>
      <td>Alone</td>
      <td>Sunny</td>
      <td>55</td>
      <td>2PM</td>
      <td>Restaurant(&lt;20)</td>
      <td>1d</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Coffee House</td>
      <td>2h</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Carry out &amp; Take away</td>
      <td>2h</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>2PM</td>
      <td>Coffee House</td>
      <td>2h</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>2PM</td>
      <td>Coffee House</td>
      <td>1d</td>
      <td>Female</td>
      <td>21</td>
      <td>Unmarried partner</td>
      <td>...</td>
      <td>never</td>
      <td>NaN</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12679</th>
      <td>Home</td>
      <td>Partner</td>
      <td>Rainy</td>
      <td>55</td>
      <td>6PM</td>
      <td>Carry out &amp; Take away</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12680</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Rainy</td>
      <td>55</td>
      <td>7AM</td>
      <td>Carry out &amp; Take away</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12681</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Snowy</td>
      <td>30</td>
      <td>7AM</td>
      <td>Coffee House</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12682</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Snowy</td>
      <td>30</td>
      <td>7AM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12683</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Sunny</td>
      <td>80</td>
      <td>7AM</td>
      <td>Restaurant(20-50)</td>
      <td>2h</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12610 rows × 26 columns</p>
</div>




```python
# find nan counts and replace them with some value
data['car'] = data['car'].fillna("no car data available")

nan_counts = data.isna().sum()
print(nan_counts)
```

    destination               0
    passanger                 0
    weather                   0
    temperature               0
    time                      0
    coupon                    0
    expiration                0
    gender                    0
    age                       0
    maritalStatus             0
    has_children              0
    education                 0
    occupation                0
    income                    0
    car                       0
    Bar                     107
    CoffeeHouse             217
    CarryAway               151
    RestaurantLessThan20    130
    Restaurant20To50        189
    toCoupon_GEQ5min          0
    toCoupon_GEQ15min         0
    toCoupon_GEQ25min         0
    direction_same            0
    direction_opp             0
    Y                         0
    dtype: int64



```python
# drop the rows where the data not available for bar, CoffeeHouse etc.
data = data.dropna()
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>destination</th>
      <th>passanger</th>
      <th>weather</th>
      <th>temperature</th>
      <th>time</th>
      <th>coupon</th>
      <th>expiration</th>
      <th>gender</th>
      <th>age</th>
      <th>maritalStatus</th>
      <th>...</th>
      <th>CoffeeHouse</th>
      <th>CarryAway</th>
      <th>RestaurantLessThan20</th>
      <th>Restaurant20To50</th>
      <th>toCoupon_GEQ5min</th>
      <th>toCoupon_GEQ15min</th>
      <th>toCoupon_GEQ25min</th>
      <th>direction_same</th>
      <th>direction_opp</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>No Urgent Place</td>
      <td>Alone</td>
      <td>Sunny</td>
      <td>55</td>
      <td>2PM</td>
      <td>Restaurant(&lt;20)</td>
      <td>1d</td>
      <td>Male</td>
      <td>21</td>
      <td>Single</td>
      <td>...</td>
      <td>less1</td>
      <td>4~8</td>
      <td>4~8</td>
      <td>less1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Coffee House</td>
      <td>2h</td>
      <td>Male</td>
      <td>21</td>
      <td>Single</td>
      <td>...</td>
      <td>less1</td>
      <td>4~8</td>
      <td>4~8</td>
      <td>less1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Male</td>
      <td>21</td>
      <td>Single</td>
      <td>...</td>
      <td>less1</td>
      <td>4~8</td>
      <td>4~8</td>
      <td>less1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Carry out &amp; Take away</td>
      <td>2h</td>
      <td>Male</td>
      <td>21</td>
      <td>Single</td>
      <td>...</td>
      <td>less1</td>
      <td>4~8</td>
      <td>4~8</td>
      <td>less1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>2PM</td>
      <td>Coffee House</td>
      <td>1d</td>
      <td>Male</td>
      <td>21</td>
      <td>Single</td>
      <td>...</td>
      <td>less1</td>
      <td>4~8</td>
      <td>4~8</td>
      <td>less1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12679</th>
      <td>Home</td>
      <td>Partner</td>
      <td>Rainy</td>
      <td>55</td>
      <td>6PM</td>
      <td>Carry out &amp; Take away</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12680</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Rainy</td>
      <td>55</td>
      <td>7AM</td>
      <td>Carry out &amp; Take away</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12681</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Snowy</td>
      <td>30</td>
      <td>7AM</td>
      <td>Coffee House</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12682</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Snowy</td>
      <td>30</td>
      <td>7AM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12683</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Sunny</td>
      <td>80</td>
      <td>7AM</td>
      <td>Restaurant(20-50)</td>
      <td>2h</td>
      <td>Male</td>
      <td>26</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>4~8</td>
      <td>1~3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12079 rows × 26 columns</p>
</div>



4. What proportion of the total observations chose to accept the coupon?




```python
# find the percentage of users who have accepted the coupon
accepted = data[data['Y'] == 1].shape[0]
rejected = data[data['Y'] == 0].shape[0]
perc_accepted = round((accepted / (accepted + rejected)) * 100)
print(perc_accepted)
```

    57


5. Use a bar plot to visualize the `coupon` column.


```python
sns.countplot(x='coupon', data=data, hue='coupon', legend=False)
plt.title('Bar Plot of Coupon Column')
plt.xticks(rotation=45)
plt.savefig('images/bar_plot_coupon.png', bbox_inches='tight')  # bbox_inches avoids label cutoff
plt.show()
```


    
![png](prompt_files/prompt_18_0.png)
    


6. Use a histogram to visualize the temperature column.


```python
sns.histplot(data=data, x="temperature")
plt.title('Histogram for Temperature data')
plt.savefig('images/hist_temperature.png', bbox_inches='tight')  # bbox_inches avoids label cutoff
plt.show()
```


    
![png](prompt_files/prompt_20_0.png)
    


**Investigating the Bar Coupons**

Now, we will lead you through an exploration of just the bar related coupons.  

1. Create a new `DataFrame` that contains just the bar coupons.



```python
df_bar_coupons = data.query('coupon == "Bar" and Y == 1')
print(type(df_bar_coupons))
df_bar_coupons
```

    <class 'pandas.core.frame.DataFrame'>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>destination</th>
      <th>passanger</th>
      <th>weather</th>
      <th>temperature</th>
      <th>time</th>
      <th>coupon</th>
      <th>expiration</th>
      <th>gender</th>
      <th>age</th>
      <th>maritalStatus</th>
      <th>...</th>
      <th>CoffeeHouse</th>
      <th>CarryAway</th>
      <th>RestaurantLessThan20</th>
      <th>Restaurant20To50</th>
      <th>toCoupon_GEQ5min</th>
      <th>toCoupon_GEQ15min</th>
      <th>toCoupon_GEQ25min</th>
      <th>direction_same</th>
      <th>direction_opp</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Male</td>
      <td>21</td>
      <td>Single</td>
      <td>...</td>
      <td>less1</td>
      <td>4~8</td>
      <td>4~8</td>
      <td>less1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Home</td>
      <td>Alone</td>
      <td>Sunny</td>
      <td>55</td>
      <td>6PM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Male</td>
      <td>21</td>
      <td>Single</td>
      <td>...</td>
      <td>less1</td>
      <td>4~8</td>
      <td>4~8</td>
      <td>less1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Work</td>
      <td>Alone</td>
      <td>Sunny</td>
      <td>55</td>
      <td>7AM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Male</td>
      <td>21</td>
      <td>Single</td>
      <td>...</td>
      <td>less1</td>
      <td>4~8</td>
      <td>4~8</td>
      <td>less1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>75</th>
      <td>No Urgent Place</td>
      <td>Kid(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10AM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Male</td>
      <td>46</td>
      <td>Married partner</td>
      <td>...</td>
      <td>1~3</td>
      <td>1~3</td>
      <td>1~3</td>
      <td>less1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Home</td>
      <td>Alone</td>
      <td>Sunny</td>
      <td>55</td>
      <td>6PM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Male</td>
      <td>46</td>
      <td>Married partner</td>
      <td>...</td>
      <td>1~3</td>
      <td>1~3</td>
      <td>1~3</td>
      <td>less1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12570</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>55</td>
      <td>10PM</td>
      <td>Bar</td>
      <td>2h</td>
      <td>Male</td>
      <td>21</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>1~3</td>
      <td>less1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12573</th>
      <td>No Urgent Place</td>
      <td>Alone</td>
      <td>Rainy</td>
      <td>55</td>
      <td>10AM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Male</td>
      <td>21</td>
      <td>Single</td>
      <td>...</td>
      <td>never</td>
      <td>1~3</td>
      <td>1~3</td>
      <td>less1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12591</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>80</td>
      <td>10PM</td>
      <td>Bar</td>
      <td>1d</td>
      <td>Female</td>
      <td>20</td>
      <td>Divorced</td>
      <td>...</td>
      <td>less1</td>
      <td>1~3</td>
      <td>1~3</td>
      <td>less1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12644</th>
      <td>No Urgent Place</td>
      <td>Friend(s)</td>
      <td>Sunny</td>
      <td>55</td>
      <td>10PM</td>
      <td>Bar</td>
      <td>2h</td>
      <td>Male</td>
      <td>31</td>
      <td>Married partner</td>
      <td>...</td>
      <td>never</td>
      <td>4~8</td>
      <td>gt8</td>
      <td>less1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12652</th>
      <td>Home</td>
      <td>Partner</td>
      <td>Sunny</td>
      <td>30</td>
      <td>10PM</td>
      <td>Bar</td>
      <td>2h</td>
      <td>Male</td>
      <td>31</td>
      <td>Married partner</td>
      <td>...</td>
      <td>never</td>
      <td>4~8</td>
      <td>gt8</td>
      <td>less1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>788 rows × 26 columns</p>
</div>



2. What proportion of bar coupons were accepted?



```python
coupons_accepted_df = data.query("Y == 1")
bar_coupons_accepted = round((df_bar_coupons.shape[0] / coupons_accepted_df.shape[0]) * 100)
bar_coupons_accepted
```




    11



3. Compare the acceptance rate between those who went to a bar 3 or fewer times a month to those who went more.



```python
#df_bar_coupons["Bar"].value_counts()
less_than_3 = ["1~3", "less1"]
greater_than_3 = ["4~8", "gt8"]

bar_accep_rate = round((df_bar_coupons.query('Bar in @less_than_3').shape[0] 
                        / (df_bar_coupons.query('Bar in @less_than_3').shape[0] 
                           + df_bar_coupons.query('Bar in @greater_than_3').shape[0])) * 100)
bar_accep_rate
```




    77



4. Compare the acceptance rate between drivers who go to a bar more than once a month and are over the age of 25 to the all others.  Is there a difference?



```python
greater_than_1 = ["1~3", "4~8", "gt8"]

drivers_4 = round((df_bar_coupons.query('Bar in @greater_than_1' 
                                        and 'age > 25').shape[0] / coupons_accepted_df.shape[0]) * 100)
drivers_4
```




    7



5. Use the same process to compare the acceptance rate between drivers who go to bars more than once a month and had passengers that were not a kid and had occupations other than farming, fishing, or forestry.



```python
passenger_not_kid = ["Friend(s)", "Partner"]
occupations_other = ["Farming Fishing & Forestry"]

drivers_5 = round((df_bar_coupons.query('Bar in @greater_than_1' 
                                        and 'passenger in @passenger_not_kid' 
                                        and 'occupation not in @occupations_other').shape[0] / coupons_accepted_df.shape[0]) * 100)
drivers_5
```




    11



6. Compare the acceptance rates between those drivers who:

- go to bars more than once a month, had passengers that were not a kid, and were not widowed *OR*
- go to bars more than once a month and are under the age of 30 *OR*
- go to cheap restaurants more than 4 times a month and income is less than 50K.




```python
marital_status_widowed = ["Widowed"]
cheap_restaurants = ["4~8", "gt8"]
income_less_50k = ["$25000 - $37499", "$12500 - $24999", "$37500 - $49999"]

count1 = df_bar_coupons.query(('Bar in @greater_than_1' 
                                        and 'passenger in @passenger_not_kid' 
                                        and 'maritalStatus not in @marital_status_widowed')).shape[0]

count2 = df_bar_coupons.query(('Bar in @greater_than_1'
                                          and 'age < 30')).shape[0]

count3 = data.query(('RestaurantLessThan20 in @cheap_restaurants'
                                          and 'income in @income_less_50k')).shape[0]

count4 = count1 + count2 + count3

drivers6 = round((count4 / coupons_accepted_df.shape[0]) * 100)
drivers6
```




    96



7.  Based on these observations, what do you hypothesize about drivers who accepted the bar coupons?


```python
print('Drivers who are driving with Friends or partner under the age of 30 with lower income grade who goes to cheaper restaurant are likely to accept the coupons.')
```

    Drivers who are driving with Friends or partner under the age of 30 with lower income grade who goes to cheaper restaurant are likely to accept the coupons.


### Independent Investigation

Using the bar coupon example as motivation, you are to explore one of the other coupon groups and try to determine the characteristics of passengers who accept the coupons.  

Identity the income of drivers accepting coupon for cheaper restaurant. Which group accepts the most?


```python
df_rest_under20_coupons = data.query('coupon == "Restaurant(<20)" and Y == 1')
#ii_df1 = df_rest_under20_coupons.groupby("income").size().reset_index(name='Count')
sns.countplot(x='income', data=df_rest_under20_coupons, hue='income', legend=False)
plt.title('Income group accepting cheaper restaurant')
plt.xticks(rotation=90)
plt.savefig('images/bar_plot_rest_under20_income.png', bbox_inches='tight')  # bbox_inches avoids label cutoff
plt.show()
```


    
![png](prompt_files/prompt_37_0.png)
    


Identity the education of drivers accepting coupon for expensive restaurant. Which group accepts the most?


```python
df_rest_2050_coupons = data.query('coupon == "Restaurant(20-50)" and Y == 1')
sns.countplot(x='income', data=df_rest_2050_coupons, hue='income', legend=False)
plt.title('Income group accepting expensive restaurant')
plt.xticks(rotation=90)
plt.savefig('images/bar_plot_rest_above20_income.png', bbox_inches='tight')  # bbox_inches avoids label cutoff
plt.show()
```


    
![png](prompt_files/prompt_39_0.png)
    


Above 2 graphs show that income has nothing to do with accepting coupons for regular or expensive restaurant.

Identify if weather condition has any impact on accepting the coupon


```python
 sns.histplot(data=data.query('Y == 1'), x="weather")
plt.title('Histogram for Weather data')
plt.savefig('images/hist_weather.png', bbox_inches='tight')  # bbox_inches avoids label cutoff
plt.show()
```


    
![png](prompt_files/prompt_42_0.png)
    


Above chart proves that coupons are more accepted when weather is sunny.


```python

```
