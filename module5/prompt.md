### Required Assignment 5.1: Will the Customer Accept the Coupon?

**Here are the list of observations and conclusions based from Jupyter Notebook**

Jupyter Notebook: [prompt.ipynb](https://github.com/shahcj/uc_ml_ai_assignments/blob/main/module5/prompt.ipynb)

**Observations:**
* The data consists of 12,684 rows and 26 columns. Based on the value counts observed, we had to make following code cleaning:
    * Rename passanger column to passenger using *rename* function.
    * Convert age column to integer using *astype(int)*. Updated 2 values **below21** to **20** and **50plus** to **50**
    * We dropped 74 duplicates rows using *drop_duplicates()* function
    * We dropped 531 rows from data with NaN values

**Conclusions:**
* The total proportion of the drivers accepting the Coupon is 57%.
* The total proportion of the bar coupons accepted are 11%.
* Drivers going to bar less were accepting coupons more.
    * 492 drivers accepted coupons who went to bar less than 3 times.
    * 147 drivers accepted coupons who went to bar more than 3 times.
    * The acceptance rate between those who went to a bar 3 or fewer times a month to those who went more is 77%.
* 8% of drivers who accepted coupons for bar went to bar more than once per month and are over age of 25.
    * 554 drivers who accepted coupons for bar went to bar more than once per month and are over age of 25.
    * 6,877 total drivers accepted the coupon for bar.
* 11% drivers accepted coupon who go to bars more than once a month and had passengers that were not a kid and had occupations other than farming, fishing, or forestry.
* All below conditions makes the majority of the drivers accepting the coupons.
    * 781 drivers who accepted the coupon go to bars more than once a month, had passengers that were not a kid, and were not widowed.
    * 413 drivers who accepted the coupon go to bars more than once a month and are under the age of 30.
    * 5,336 drivers who accepted the coupon go to cheap restaurants more than 4 times a month and income is less than 50K.
    * 95% drivers accepted coupon with above listed conditions.
* **Hypothesize**: Drivers who are driving with Friends or partner under the age of 30 with lower income grade who goes to cheaper restaurant are likely to accept the coupons.
* **Independent Investigation**
    * Looks like variety of income groups accepts the coupons. Income does not influence the choice.
    * Looks like variety of income groups goes to expensive restaurants too.
    * Coupons are more accepted when weather is sunny.

