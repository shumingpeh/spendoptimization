
___
due to sensitive information, actual real numbers will not be used here. instead it will be substituted with real stocks

This notebook will show the concept of spend optimization based on returns and variance. It will answer the question of:
1. if you have $100, where will you spend it on?
1. the spend allocation should give us the maximum returns and lowest variance


```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime 
from dateutil.relativedelta import relativedelta
import cvxopt as opt
from cvxopt import blas, solvers, matrix
```

### Context of intention of analysis
- Lets say we have `$100`, where should we spend on paid acquisition of users?
- So that ROI will be more than current and also take on lesser risk
- This becomes a 'portfolio' optimization problem, and each channel and geo can be considered as 1 stock. We will end up with a lot of options to choose from
    - Lets use `YouTube` for example. `YouTube` singapore will be considered as 1 stock
- And we will choose a bunch of stocks that will return us the maximum returns and lowest variance
- Here, we will not be using the real data i had, but instead use real stock information to illustrate the idea

### Scope of analysis
- Data pull
- Generate covariance and correlation matrix
- Pick stocks based on optimization
- Get portfolio return and variance (and sharpe ratio)
- Plot efficient frontier

### Data pull
- the data used will be from the previous blog post of k-means: [link](https://medium.com/uptick-blog/stock-picks-using-k-means-clustering-4330c6c4e8de)
- we will be focusing on the golden cluster identified, which has 57 stocks with an average of ~24% return and ~5% variance


```python
rawdata = (
    pd.read_csv('data/aggregate_tickers_return_variance.csv', index_col=0)
    .reset_index()
)

rawdata_yearly_returns = (
    pd.read_csv("data/yearly_returns_for_tickers.csv")
)
```

#### Query the golden cluster stocks only


```python
golden_cluster_stocks = (
    rawdata
    .query("ticker in ('ADSK','ANTM', 'AZPN', 'CACC', 'CHDN', 'CHE', 'CI', 'COO', 'DENN',\
       'FDEF', 'FICO', 'FIS', 'FIX', 'HELE', 'HIFS', 'ICUI', 'IHC',\
       'INTU', 'IT', 'JKHY', 'KAI', 'KWR', 'LFUS', 'LII', 'LION', 'LOW',\
       'MCO', 'MLAB', 'MMSI', 'MOH', 'MSFT', 'MTD', 'MTN', 'NSP', 'OLBK',\
       'OMCL', 'ORLY', 'PRSC', 'RDI', 'RHT', 'RMD', 'SBAC', 'SCHW', 'SCI',\
       'SHEN', 'SHW', 'STE', 'TDY', 'TFX', 'TMO', 'TSS', 'ULTI', 'UNH',\
       'USPH', 'VLO', 'VRSN', 'ZBRA')")
)
```

### Maximum returns and variance + mininum returns and variance of stocks
- this sets the maximum and minimum boundary of both returns and variance of portfolio


```python
max_return = golden_cluster_stocks.avg_yearly_returns.max()
min_return = golden_cluster_stocks.avg_yearly_returns.min()
min_variance = golden_cluster_stocks.yearly_variance.min()
max_variance = golden_cluster_stocks.yearly_variance.max()

portfolio_boundary = {'max_return': [max_return],
        'min_return': [min_return],
        'max_variance': [max_variance],
        'min_variance': [min_variance]}
portfolio_boundary_df = (
    pd.DataFrame(portfolio_boundary, columns=['max_return', 'min_return', 'max_variance', 'min_variance'])
    .pipe(lambda x:x.assign(max_stdev = np.sqrt(x.max_variance)))
    .pipe(lambda x:x.assign(min_stdev = np.sqrt(x.min_variance)))
)
portfolio_boundary_df
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
      <th>max_return</th>
      <th>min_return</th>
      <th>max_variance</th>
      <th>min_variance</th>
      <th>max_stdev</th>
      <th>min_stdev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.306971</td>
      <td>0.199287</td>
      <td>0.101233</td>
      <td>0.009517</td>
      <td>0.318171</td>
      <td>0.097555</td>
    </tr>
  </tbody>
</table>
</div>



### Covariance matrix
- diagonals are variance of stocks
- Covariance is a measure of the directional relationship between the returns on two assets.
    - Positive covariance means that both returns move together while a negative covariance means returns move inversely


```python
golden_cluster_covariance_df = (
    rawdata_yearly_returns
    .drop(['year'],1)
    .cov()
)

golden_cluster_covariance_df.head()
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
      <th>ADSK</th>
      <th>ANTM</th>
      <th>AZPN</th>
      <th>CACC</th>
      <th>CHDN</th>
      <th>CHE</th>
      <th>CI</th>
      <th>COO</th>
      <th>DENN</th>
      <th>FDEF</th>
      <th>...</th>
      <th>TDY</th>
      <th>TFX</th>
      <th>TMO</th>
      <th>TSS</th>
      <th>ULTI</th>
      <th>UNH</th>
      <th>USPH</th>
      <th>VLO</th>
      <th>VRSN</th>
      <th>ZBRA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ADSK</th>
      <td>0.019131</td>
      <td>0.027225</td>
      <td>0.007918</td>
      <td>-0.002667</td>
      <td>0.002815</td>
      <td>-0.003398</td>
      <td>0.013265</td>
      <td>0.016658</td>
      <td>0.009122</td>
      <td>0.003296</td>
      <td>...</td>
      <td>0.026112</td>
      <td>0.016444</td>
      <td>0.018055</td>
      <td>0.014340</td>
      <td>0.003236</td>
      <td>0.013573</td>
      <td>-0.009848</td>
      <td>-0.002619</td>
      <td>0.007979</td>
      <td>0.015531</td>
    </tr>
    <tr>
      <th>ANTM</th>
      <td>0.027225</td>
      <td>0.061648</td>
      <td>-0.019889</td>
      <td>0.012339</td>
      <td>0.017044</td>
      <td>0.009364</td>
      <td>0.041132</td>
      <td>0.010838</td>
      <td>0.004594</td>
      <td>-0.001503</td>
      <td>...</td>
      <td>0.024969</td>
      <td>0.029038</td>
      <td>0.027257</td>
      <td>0.042327</td>
      <td>0.006281</td>
      <td>0.024942</td>
      <td>-0.025032</td>
      <td>-0.001225</td>
      <td>0.034282</td>
      <td>0.019255</td>
    </tr>
    <tr>
      <th>AZPN</th>
      <td>0.007918</td>
      <td>-0.019889</td>
      <td>0.066590</td>
      <td>-0.006476</td>
      <td>0.002020</td>
      <td>-0.024152</td>
      <td>0.000837</td>
      <td>0.016797</td>
      <td>0.009201</td>
      <td>0.015412</td>
      <td>...</td>
      <td>0.026970</td>
      <td>0.000812</td>
      <td>0.027584</td>
      <td>0.001465</td>
      <td>0.029782</td>
      <td>-0.004612</td>
      <td>0.012578</td>
      <td>0.028989</td>
      <td>-0.002318</td>
      <td>-0.006798</td>
    </tr>
    <tr>
      <th>CACC</th>
      <td>-0.002667</td>
      <td>0.012339</td>
      <td>-0.006476</td>
      <td>0.041870</td>
      <td>0.039154</td>
      <td>0.021414</td>
      <td>0.038505</td>
      <td>-0.025871</td>
      <td>-0.033232</td>
      <td>-0.018230</td>
      <td>...</td>
      <td>-0.010575</td>
      <td>0.010962</td>
      <td>0.010513</td>
      <td>0.046065</td>
      <td>0.023787</td>
      <td>-0.004092</td>
      <td>-0.011748</td>
      <td>0.035648</td>
      <td>0.048689</td>
      <td>-0.029817</td>
    </tr>
    <tr>
      <th>CHDN</th>
      <td>0.002815</td>
      <td>0.017044</td>
      <td>0.002020</td>
      <td>0.039154</td>
      <td>0.043083</td>
      <td>0.022508</td>
      <td>0.047096</td>
      <td>-0.015737</td>
      <td>-0.027221</td>
      <td>-0.007936</td>
      <td>...</td>
      <td>0.000777</td>
      <td>0.019675</td>
      <td>0.018952</td>
      <td>0.050159</td>
      <td>0.027737</td>
      <td>0.001630</td>
      <td>-0.015988</td>
      <td>0.047558</td>
      <td>0.043300</td>
      <td>-0.031412</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 57 columns</p>
</div>



### Correlation matrix
- $\rho = \frac{\sigma_{xy}}{\sigma_x \sigma_y} = \frac{covariance}{\sigma_x \sigma_y}
$
- this tells us how each stock return correlate with each other
- ideally, we would like our portfolio to have a good balance of negative and positive correlation


```python
sigma_x_sigma_y = (
#     pd.DataFrame(
        np.dot(
            (np.array([np.sqrt(golden_cluster_stocks.yearly_variance)]).T),
            np.array([np.sqrt(golden_cluster_stocks.yearly_variance)])
        )
#     )
)
```


```python
golden_cluster_correlation_df = golden_cluster_covariance_df/sigma_x_sigma_y
golden_cluster_correlation_df.head()
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
      <th>ADSK</th>
      <th>ANTM</th>
      <th>AZPN</th>
      <th>CACC</th>
      <th>CHDN</th>
      <th>CHE</th>
      <th>CI</th>
      <th>COO</th>
      <th>DENN</th>
      <th>FDEF</th>
      <th>...</th>
      <th>TDY</th>
      <th>TFX</th>
      <th>TMO</th>
      <th>TSS</th>
      <th>ULTI</th>
      <th>UNH</th>
      <th>USPH</th>
      <th>VLO</th>
      <th>VRSN</th>
      <th>ZBRA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ADSK</th>
      <td>1.000000</td>
      <td>0.792779</td>
      <td>0.221850</td>
      <td>-0.094226</td>
      <td>0.098064</td>
      <td>-0.143952</td>
      <td>0.352907</td>
      <td>0.667336</td>
      <td>0.343442</td>
      <td>0.131142</td>
      <td>...</td>
      <td>0.881483</td>
      <td>0.725899</td>
      <td>0.549575</td>
      <td>0.385130</td>
      <td>0.098097</td>
      <td>0.705511</td>
      <td>-0.555879</td>
      <td>-0.063110</td>
      <td>0.207719</td>
      <td>0.534096</td>
    </tr>
    <tr>
      <th>ANTM</th>
      <td>0.792779</td>
      <td>1.000000</td>
      <td>-0.310426</td>
      <td>0.242863</td>
      <td>0.330721</td>
      <td>0.220977</td>
      <td>0.609572</td>
      <td>0.241871</td>
      <td>0.096340</td>
      <td>-0.033316</td>
      <td>...</td>
      <td>0.469554</td>
      <td>0.714042</td>
      <td>0.462175</td>
      <td>0.633268</td>
      <td>0.106084</td>
      <td>0.722206</td>
      <td>-0.787152</td>
      <td>-0.016445</td>
      <td>0.497138</td>
      <td>0.368871</td>
    </tr>
    <tr>
      <th>AZPN</th>
      <td>0.221850</td>
      <td>-0.310426</td>
      <td>1.000000</td>
      <td>-0.122652</td>
      <td>0.037707</td>
      <td>-0.548409</td>
      <td>0.011938</td>
      <td>0.360665</td>
      <td>0.185666</td>
      <td>0.328712</td>
      <td>...</td>
      <td>0.487998</td>
      <td>0.019201</td>
      <td>0.450030</td>
      <td>0.021087</td>
      <td>0.483980</td>
      <td>-0.128493</td>
      <td>0.380544</td>
      <td>0.374423</td>
      <td>-0.032339</td>
      <td>-0.125297</td>
    </tr>
    <tr>
      <th>CACC</th>
      <td>-0.094226</td>
      <td>0.242863</td>
      <td>-0.122652</td>
      <td>1.000000</td>
      <td>0.921865</td>
      <td>0.613198</td>
      <td>0.692414</td>
      <td>-0.700537</td>
      <td>-0.845726</td>
      <td>-0.490346</td>
      <td>...</td>
      <td>-0.241317</td>
      <td>0.327079</td>
      <td>0.216300</td>
      <td>0.836278</td>
      <td>0.487479</td>
      <td>-0.143783</td>
      <td>-0.448265</td>
      <td>0.580653</td>
      <td>0.856738</td>
      <td>-0.693110</td>
    </tr>
    <tr>
      <th>CHDN</th>
      <td>0.098064</td>
      <td>0.330721</td>
      <td>0.037707</td>
      <td>0.921865</td>
      <td>1.000000</td>
      <td>0.635396</td>
      <td>0.834894</td>
      <td>-0.420103</td>
      <td>-0.682917</td>
      <td>-0.210443</td>
      <td>...</td>
      <td>0.017480</td>
      <td>0.578727</td>
      <td>0.384392</td>
      <td>0.897691</td>
      <td>0.560374</td>
      <td>0.056466</td>
      <td>-0.601386</td>
      <td>0.763664</td>
      <td>0.751099</td>
      <td>-0.719840</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 57 columns</p>
</div>



### Pick stocks based on optimization

#### Idea of portfolio optimization
- Ideally, we would want our portfolio to have high returns and low risk (standard deviation)
- Assuming we have a list of stocks to pick, with historical information of returns and risk, is it possible to pick stocks with different weightage to have returns higher than average and risk lower than average?
- Yes, markowitz mean-variance portfolio optimization allows us to achieve just the above
- Intuitively without any mathematical formula introduced, we would like to: 
    1. minimize variance 
    1. achieve a set amount of returns
    1. sum of weights of all stock = 1
    1. we assume no short position, so weightage of one stock >= 0 and <= 1
    
    
- Now if we have to place the objective function, it will be the portfolio standard deviation:
    - $ \sigma_p = \sqrt{\sum_{i}\sum_{j}w_{i}w_{j}\sigma_{ij}} $ = $ w^T * Cov_{matrix} * w$
- subjected to the following constraints:
    - portfolio returns = $ \sum (w_i *\mu_i) $ = x
    - sum of weights = $\sum w_i$ = 1
    - weight of each stock = $ 0 \leq w_i \leq 1$
        - $\mu_i$: expected return of stock
        - $w_i$: allocation weight of stock
        - $x$: target of expected return
        - $\sigma_{ij}$: covariance of stock returns between 2 stocks


- if we think a little more about the covariance formula, $ \frac{1}{n} \sum((x_i - \mu_x)(y_i - \mu_y))$
    - it is a non-linear equation but more specifically a quadratic equation
    - which means there is a specific solution if constraints are fulfilled
    - we can go about solving this with the convex approach


```python
golden_cluster_covariance_matrix = matrix(golden_cluster_covariance_df.as_matrix(),(57,57))
```

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:1: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      """Entry point for launching an IPython kernel.



```python
golden_cluster_return_matrix = (
    matrix(
        golden_cluster_stocks
        [['avg_yearly_returns']]
        .as_matrix(), (57,1)
    )
)
```

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:4: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      after removing the cwd from sys.path.


### Initialize optimization function for weights allocation


```python
# Input:  
# n: number of assets
# avg_ret: nx1 matrix of average returns
# covs: nxn matrix of covariance
# r_min: the minimum expected return to achieve
def optimize_portfolio(n, avg_ret, covs, r_min):
    P = covs
    q = matrix(np.zeros((n, 1)), tc='d')
    # constraints (avg_ret'x >= r_min) and (x >= 0)
    G = matrix(np.concatenate((
        -np.transpose(np.array(avg_ret)), 
        -np.identity(n)), 0))
    h = matrix(np.concatenate((
        -np.ones((1,1))*r_min, 
        np.zeros((n,1))), 0))
    # equality constraint Ax = b
    #fulfills the constraint sum(x) == 1
    A = matrix(1.0, (1,n))
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)
    return sol
```

### Get weights of portfolio
- `r_min` is set at 15% higher than the default average of golden cluster
- hopefully the variance will be lower than the average of the golden cluster


```python
solution = optimize_portfolio(57, golden_cluster_return_matrix, golden_cluster_covariance_matrix, 0.290)
```

         pcost       dcost       gap    pres   dres
     0:  2.1153e-03 -9.6139e-01  6e+01  8e+00  1e+01
     1:  3.4436e-03 -9.2464e-01  4e+00  4e-01  5e-01
     2:  3.4483e-03 -2.6624e-01  2e+00  2e-01  2e-01
     3:  1.2769e-02 -4.8001e-01  1e+00  4e-02  5e-02
     4:  1.3740e-02 -1.1873e-01  1e-01  6e-04  7e-04
     5:  1.3494e-02 -4.0676e-03  2e-02  7e-05  9e-05
     6:  6.1668e-03 -1.5405e-02  2e-02  3e-05  4e-05
     7:  5.6270e-03  2.1773e-03  3e-03  5e-06  6e-06
     8:  4.7511e-03  2.2114e-03  3e-03  1e-06  1e-06
     9:  4.0740e-03  3.7644e-03  3e-04  1e-07  1e-07
    10:  3.9294e-03  3.8560e-03  7e-05  2e-08  2e-08
    11:  3.8789e-03  3.8755e-03  3e-06  1e-16  7e-13
    12:  3.8760e-03  3.8759e-03  7e-08  2e-16  2e-12
    Optimal solution found.


### Dataframe of portfolio weights


```python
portfolio_weights = (
    pd.DataFrame(np.array(solution['x']))
    .pipe(lambda x:x.assign(ticker = golden_cluster_stocks.ticker.values))
    .rename(columns={0:"weightage"})
#     .sort_values(['weightage'],ascending=False)
    .pipe(lambda x:x.assign(nominal_amount_every_1000 = x.weightage*1000))
    .merge(golden_cluster_stocks,how='left',on=['ticker'])
    .pipe(lambda x:x.assign(default_weights = 1/57))
    .pipe(lambda x:x.assign(weightage = round(x.weightage,4)))
#     .pipe(lambda x:x.assign(cumsum_weightage = x.weightage.cumsum()))
)
```

### Portfolio returns, standard deviation and sharpe ratio
- sharpe ratio is a measure of risk-to-reward ratio, $sharpeRatio = \frac{portfolioReturns}{portfolioStdDev}$
    - a higher magnitude the better it is


```python
portfolio_weights.dtypes
```




    weightage                    float64
    ticker                        object
    nominal_amount_every_1000    float64
    avg_yearly_returns           float64
    yearly_variance              float64
    default_weights              float64
    dtype: object




```python
portfolio_returns = np.dot(
    matrix(portfolio_weights[['weightage']].as_matrix(), (1,57)),
    matrix(golden_cluster_stocks[['avg_yearly_returns']].as_matrix(), (57,1))
)
```

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:3: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      This is separate from the ipykernel package so we can avoid doing imports until



```python
temp_value = np.dot(
        matrix(portfolio_weights[['weightage']].as_matrix(), (1,57)),
        golden_cluster_covariance_matrix
)

portfolio_stdev = np.sqrt(np.dot(temp_value,matrix(portfolio_weights[['weightage']].as_matrix(), (57,1))))

portfolio_sharpe_ratio = portfolio_returns/portfolio_stdev
```

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:6: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
      



```python
print("portfolio returns: " + str(round(float(portfolio_returns),4)))
print("portfolio standard deviation: " + str(round(float((portfolio_stdev)),4))) 
print("portfolio sharpe ratio: " + str(round(float((portfolio_sharpe_ratio)),4))) 
```

    portfolio returns: 0.29
    portfolio standard deviation: 0.088
    portfolio sharpe ratio: 3.294



```python
portfolio_weights_sort_weight = (
    portfolio_weights
    .sort_values(['weightage'],ascending=False)
    .pipe(lambda x:x.assign(cumsum_weightage = x.weightage.cumsum()))
)
```

### Construct efficient frontier
- assuming we optimize the returns for each possible scenario


```python
def efficient_frontier(returns):
    temp_solution = optimize_portfolio(57, golden_cluster_return_matrix, golden_cluster_covariance_matrix, returns)
    portfolio_weights = (
        pd.DataFrame(np.array(temp_solution['x']))
        .pipe(lambda x:x.assign(ticker = golden_cluster_stocks.ticker.values))
        .rename(columns={0:"weightage"})
    #     .sort_values(['weightage'],ascending=False)
        .pipe(lambda x:x.assign(nominal_amount_every_1000 = x.weightage*1000))
        .merge(golden_cluster_stocks,how='left',on=['ticker'])
        .pipe(lambda x:x.assign(default_weights = 1/57))
        .pipe(lambda x:x.assign(weightage = round(x.weightage,10)))
    #     .pipe(lambda x:x.assign(cumsum_weightage = x.weightage.cumsum()))
    )
    
    if returns != 0:
        portfolio_returns = np.dot(
            matrix(portfolio_weights[['weightage']].as_matrix(), (1,57)),
            matrix(golden_cluster_stocks[['avg_yearly_returns']].as_matrix(), (57,1))
        )

        temp_value = np.dot(
            matrix(portfolio_weights[['weightage']].as_matrix(), (1,57)),
            golden_cluster_covariance_matrix
        )

        portfolio_stdev = np.sqrt(np.dot(temp_value,matrix(portfolio_weights[['weightage']].as_matrix(), (57,1))))

        portfolio_sharpe_ratio = portfolio_returns/portfolio_stdev

        temp_portfolio_df = pd.DataFrame(
            {'portfolio_return': [float(portfolio_returns)],
            'portfolio_stdev': [float(portfolio_stdev)],
            'portfolio_sharpe_ratio': [float(portfolio_sharpe_ratio)]},
            columns=['portfolio_return', 'portfolio_stdev', 'portfolio_sharpe_ratio']
        )
    elif returns == 0:
        portfolio_returns = np.dot(
            matrix(portfolio_weights[['default_weights']].as_matrix(), (1,57)),
            matrix(golden_cluster_stocks[['avg_yearly_returns']].as_matrix(), (57,1))
        )

        temp_value = np.dot(
            matrix(portfolio_weights[['default_weights']].as_matrix(), (1,57)),
            golden_cluster_covariance_matrix
        )

        portfolio_stdev = np.sqrt(np.dot(temp_value,matrix(portfolio_weights[['default_weights']].as_matrix(), (57,1))))

        portfolio_sharpe_ratio = portfolio_returns/portfolio_stdev

        temp_portfolio_df = pd.DataFrame(
            {'portfolio_return': [float(portfolio_returns)],
            'portfolio_stdev': [float(portfolio_stdev)],
            'portfolio_sharpe_ratio': [float(portfolio_sharpe_ratio)]},
            columns=['portfolio_return', 'portfolio_stdev', 'portfolio_sharpe_ratio']
        )
    
    return temp_portfolio_df
    

```


```python
aggregate_efficient_frontier = None
for i in np.arange(0.23, max_return, 0.001):
    if aggregate_efficient_frontier is None:
        aggregate_efficient_frontier = efficient_frontier(i)
    else:
        aggregate_efficient_frontier = aggregate_efficient_frontier.append(efficient_frontier(i))
aggregate_efficient_frontier = (
    aggregate_efficient_frontier.append(efficient_frontier(0))
    .sort_values(['portfolio_return'])
    .pipe(lambda x:x.assign(reject_portfolio_allocation = np.where(x.portfolio_sharpe_ratio < 2.61547,1,0)))
    .query("portfolio_return < 0.24 | portfolio_return >= 0.275")
)
```

         pcost       dcost       gap    pres   dres
     0:  2.1625e-03 -1.0252e+00  1e+00  2e-16  1e+01
     1:  2.1039e-03 -2.4306e-02  3e-02  1e-16  2e-01
     2:  7.8366e-04 -5.0226e-03  6e-03  2e-16  2e-02
     3:  9.0810e-05 -8.6658e-04  1e-03  4e-17  8e-04
     4:  2.0318e-06 -1.0288e-04  1e-04  2e-16  4e-05
     5:  2.7960e-10 -1.2505e-06  1e-06  4e-17  2e-07
     6:  2.7965e-14 -1.2504e-08  1e-08  9e-17  2e-09
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1617e-03 -1.0242e+00  1e+00  1e-16  1e+01
     1:  2.1032e-03 -2.4278e-02  3e-02  6e-17  2e-01
     2:  7.8304e-04 -5.0087e-03  6e-03  2e-16  2e-02
     3:  8.9886e-05 -8.6327e-04  1e-03  2e-16  8e-04
     4:  2.0318e-06 -1.0276e-04  1e-04  9e-17  4e-05
     5:  2.7879e-10 -1.2485e-06  1e-06  1e-16  2e-07
     6:  2.7885e-14 -1.2483e-08  1e-08  2e-16  2e-09
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1608e-03 -1.0232e+00  1e+00  0e+00  1e+01
     1:  2.1024e-03 -2.4250e-02  3e-02  3e-17  2e-01
     2:  7.8223e-04 -4.9914e-03  6e-03  4e-17  2e-02
     3:  8.8716e-05 -8.5921e-04  9e-04  2e-16  7e-04
     4:  2.0219e-06 -1.0244e-04  1e-04  2e-16  4e-05
     5:  2.7801e-10 -1.2462e-06  1e-06  2e-16  2e-07
     6:  2.7806e-14 -1.2460e-08  1e-08  2e-16  2e-09
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1600e-03 -1.0222e+00  1e+00  0e+00  1e+01
     1:  2.1016e-03 -2.4223e-02  3e-02  3e-17  2e-01
     2:  7.8119e-04 -4.9694e-03  6e-03  1e-16  2e-02
     3:  8.7219e-05 -8.5423e-04  9e-04  4e-16  7e-04
     4:  1.9821e-06 -1.0157e-04  1e-04  2e-16  3e-05
     5:  2.7635e-10 -1.2415e-06  1e-06  5e-17  2e-07
     6:  2.7639e-14 -1.2413e-08  1e-08  2e-16  2e-09
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1592e-03 -1.0212e+00  1e+00  0e+00  1e+01
     1:  2.1008e-03 -2.4198e-02  3e-02  3e-17  2e-01
     2:  7.7980e-04 -4.9408e-03  6e-03  5e-17  2e-02
     3:  8.8893e-05 -8.8972e-04  1e-03  4e-16  8e-04
     4:  2.3565e-06 -1.1127e-04  1e-04  4e-17  5e-05
     5:  3.4480e-10 -1.4150e-06  1e-06  2e-16  2e-07
     6:  3.4489e-14 -1.4149e-08  1e-08  2e-16  2e-09
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1584e-03 -1.0201e+00  1e+00  0e+00  1e+01
     1:  2.1001e-03 -2.4174e-02  3e-02  3e-17  2e-01
     2:  7.7791e-04 -4.9023e-03  6e-03  4e-17  2e-02
     3:  9.3176e-05 -9.5874e-04  1e-03  9e-17  1e-03
     4:  2.9941e-06 -1.2695e-04  1e-04  2e-16  9e-05
     5:  4.9792e-10 -1.7587e-06  2e-06  1e-16  2e-07
     6:  4.9815e-14 -1.7585e-08  2e-08  2e-16  2e-09
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1576e-03 -1.0191e+00  1e+00  2e-16  1e+01
     1:  2.0993e-03 -2.4152e-02  3e-02  1e-16  2e-01
     2:  7.7522e-04 -4.8483e-03  6e-03  2e-16  2e-02
     3:  9.8883e-05 -1.0479e-03  1e-03  4e-17  2e-03
     4:  3.7135e-06 -1.4390e-04  1e-04  1e-16  1e-04
     5:  8.1122e-10 -2.3093e-06  2e-06  7e-17  2e-07
     6:  8.1187e-14 -2.3093e-08  2e-08  2e-16  2e-09
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1568e-03 -1.0181e+00  1e+00  0e+00  1e+01
     1:  2.0986e-03 -2.4136e-02  3e-02  4e-17  2e-01
     2:  7.7115e-04 -4.7672e-03  6e-03  4e-17  2e-02
     3:  1.0623e-04 -1.1612e-03  1e-03  2e-16  2e-03
     4:  4.5673e-06 -1.6274e-04  2e-04  7e-17  2e-04
     5:  1.3644e-09 -3.0701e-06  3e-06  2e-16  1e-07
     6:  1.3664e-13 -3.0706e-08  3e-08  3e-17  1e-09
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1560e-03 -1.0171e+00  1e+00  0e+00  1e+01
     1:  2.0979e-03 -2.4128e-02  3e-02  1e-16  2e-01
     2:  7.6436e-04 -4.6318e-03  5e-03  2e-16  2e-02
     3:  1.1423e-04 -1.2951e-03  1e-03  5e-17  2e-03
     4:  5.5201e-06 -1.8243e-04  2e-04  5e-17  2e-04
     5:  2.3335e-09 -4.0958e-06  4e-06  5e-17  6e-18
     6:  2.3396e-13 -4.0977e-08  4e-08  1e-16  7e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1552e-03 -1.0161e+00  1e+00  0e+00  1e+01
     1:  2.0972e-03 -2.4142e-02  3e-02  9e-17  2e-01
     2:  7.5070e-04 -4.3569e-03  5e-03  4e-17  2e-02
     3:  1.1448e-04 -1.3997e-03  2e-03  1e-16  2e-03
     4:  5.9174e-06 -1.9362e-04  2e-04  2e-16  2e-04
     5:  3.1532e-09 -4.8136e-06  5e-06  8e-17  7e-18
     6:  3.1648e-13 -4.8174e-08  5e-08  6e-17  7e-18
    Optimal solution found.

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:17: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:18: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:22: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:26: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.


    
         pcost       dcost       gap    pres   dres
     0:  2.1544e-03 -1.0150e+00  1e+00  4e-16  1e+01
     1:  2.0967e-03 -2.4497e-02  3e-02  4e-17  3e-01
     2:  7.7363e-04 -4.7138e-03  5e-03  2e-16  2e-02
     3:  1.0150e-04 -1.4648e-03  2e-03  4e-17  1e-03
     4:  5.5110e-06 -1.9731e-04  2e-04  5e-17  1e-04
     5:  3.3020e-09 -4.9039e-06  5e-06  2e-16  1e-07
     6:  3.3148e-13 -4.9081e-08  5e-08  3e-16  1e-09
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1536e-03 -1.0140e+00  6e+01  8e+00  1e+01
     1:  3.5966e-03 -9.7874e-01  2e+00  7e-02  9e-02
     2:  3.6713e-03 -1.2460e-01  1e-01  9e-17  1e-15
     3:  2.9902e-03 -1.6463e-02  2e-02  6e-17  2e-15
     4:  5.2792e-04 -1.0566e-02  1e-02  9e-17  2e-15
     5:  1.5796e-04 -2.0963e-03  2e-03  7e-17  4e-16
     6:  1.0189e-05 -2.8874e-04  3e-04  2e-16  3e-17
     7:  1.4231e-08 -1.0247e-05  1e-05  7e-17  9e-18
     8:  1.4498e-12 -1.0310e-07  1e-07  6e-17  1e-17
     9:  1.4504e-16 -1.0310e-09  1e-09  4e-17  8e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1528e-03 -1.0130e+00  6e+01  8e+00  1e+01
     1:  3.5934e-03 -9.7778e-01  2e+00  8e-02  1e-01
     2:  3.6786e-03 -1.5048e-01  2e-01  6e-04  8e-04
     3:  3.0921e-03 -1.6998e-02  2e-02  8e-05  1e-04
     4:  5.4834e-04 -1.1057e-02  1e-02  1e-16  2e-14
     5:  1.4940e-04 -2.1126e-03  2e-03  4e-17  3e-15
     6:  1.0704e-05 -3.1642e-04  3e-04  9e-17  3e-17
     7:  3.0633e-08 -1.3978e-05  1e-05  7e-17  9e-18
     8:  3.1358e-12 -1.4095e-07  1e-07  6e-17  9e-18
     9:  3.1342e-16 -1.4095e-09  1e-09  2e-16  8e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1520e-03 -1.0120e+00  6e+01  8e+00  1e+01
     1:  3.5902e-03 -9.7681e-01  2e+00  8e-02  1e-01
     2:  3.6804e-03 -2.3737e-01  3e-01  7e-03  9e-03
     3:  3.2848e-03 -1.8332e-02  2e-02  6e-04  8e-04
     4:  5.9193e-04 -1.1530e-02  1e-02  3e-17  2e-14
     5:  1.4586e-04 -2.0954e-03  2e-03  4e-16  2e-15
     6:  1.1192e-05 -3.3679e-04  3e-04  4e-16  3e-17
     7:  4.8848e-08 -1.7817e-05  2e-05  4e-16  8e-18
     8:  5.0661e-12 -1.8061e-07  2e-07  8e-17  8e-18
     9:  5.0632e-16 -1.8060e-09  2e-09  4e-16  9e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1512e-03 -1.0109e+00  6e+01  8e+00  1e+01
     1:  3.5870e-03 -9.7584e-01  2e+00  9e-02  1e-01
     2:  3.6812e-03 -3.1110e-01  4e-01  1e-02  2e-02
     3:  3.3672e-03 -1.7202e-02  2e-02  6e-04  8e-04
     4:  4.7762e-04 -9.9666e-03  1e-02  1e-16  3e-14
     5:  1.3545e-04 -1.7095e-03  2e-03  4e-17  5e-15
     6:  8.1358e-06 -2.8387e-04  3e-04  2e-16  6e-17
     7:  1.7344e-08 -1.0239e-05  1e-05  4e-17  1e-17
     8:  1.7557e-12 -1.0280e-07  1e-07  2e-16  7e-18
     9:  1.7569e-16 -1.0280e-09  1e-09  1e-16  7e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1504e-03 -1.0099e+00  6e+01  8e+00  1e+01
     1:  3.5838e-03 -9.7486e-01  2e+00  1e-01  1e-01
     2:  3.6810e-03 -3.7410e-01  4e-01  2e-02  3e-02
     3:  3.4204e-03 -2.4730e-02  3e-02  6e-04  8e-04
     4:  1.6649e-03 -1.3283e-02  1e-02  3e-04  3e-04
     5:  3.2853e-04 -2.1984e-03  3e-03  5e-17  1e-14
     6:  1.6203e-05 -3.4833e-04  4e-04  2e-16  6e-16
     7:  4.3990e-08 -1.9017e-05  2e-05  4e-16  9e-18
     8:  4.6464e-12 -1.9374e-07  2e-07  4e-16  8e-18
     9:  4.6457e-16 -1.9373e-09  2e-09  2e-16  8e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1496e-03 -1.0089e+00  6e+01  8e+00  1e+01
     1:  3.5806e-03 -9.7387e-01  2e+00  1e-01  1e-01
     2:  3.6798e-03 -4.2823e-01  5e-01  3e-02  3e-02
     3:  3.4695e-03 -5.8541e-02  6e-02  1e-03  2e-03
     4:  2.0989e-03 -1.1823e-02  1e-02  3e-04  4e-04
     5:  1.0757e-03 -8.5742e-03  1e-02  1e-04  2e-04
     6:  1.9111e-04 -1.3753e-03  2e-03  1e-05  1e-05
     7:  5.4401e-06 -2.1436e-04  2e-04  2e-16  4e-16
     8:  5.4901e-09 -5.7899e-06  6e-06  1e-16  1e-17
     9:  5.5237e-13 -5.8002e-08  6e-08  7e-17  9e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1488e-03 -1.0078e+00  6e+01  8e+00  1e+01
     1:  3.5775e-03 -9.7288e-01  2e+00  1e-01  1e-01
     2:  3.6777e-03 -4.7492e-01  6e-01  3e-02  4e-02
     3:  3.5115e-03 -1.0499e-01  1e-01  2e-03  3e-03
     4:  2.5958e-03 -1.4768e-02  2e-02  4e-04  4e-04
     5:  1.8022e-03 -1.2889e-02  1e-02  3e-04  3e-04
     6:  5.7319e-04 -3.6790e-03  4e-03  6e-05  7e-05
     7:  5.1898e-05 -7.2170e-04  8e-04  5e-07  6e-07
     8:  5.4293e-07 -5.7769e-05  6e-05  9e-09  1e-08
     9:  6.8988e-11 -6.3883e-07  6e-07  8e-11  1e-10
    10:  6.8992e-15 -6.3877e-09  6e-09  8e-13  1e-12
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1480e-03 -1.0068e+00  6e+01  8e+00  1e+01
     1:  3.5743e-03 -9.7188e-01  2e+00  1e-01  2e-01
     2:  3.6748e-03 -5.1533e-01  7e-01  4e-02  5e-02
     3:  3.5471e-03 -1.5385e-01  2e-01  4e-03  4e-03
     4:  2.8915e-03 -1.6530e-02  2e-02  4e-04  5e-04
     5:  1.3459e-03 -1.2472e-02  1e-02  2e-04  3e-04
     6:  5.1583e-04 -3.4819e-03  4e-03  5e-05  6e-05
     7:  4.7341e-05 -6.8433e-04  7e-04  5e-17  6e-16
     8:  3.4450e-07 -4.7267e-05  5e-05  2e-16  6e-17
     9:  4.2416e-11 -5.1320e-07  5e-07  1e-16  8e-18
    10:  4.2418e-15 -5.1316e-09  5e-09  2e-16  9e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1472e-03 -1.0057e+00  6e+01  8e+00  1e+01
     1:  3.5711e-03 -9.7088e-01  2e+00  1e-01  2e-01
     2:  3.6712e-03 -5.5040e-01  8e-01  5e-02  6e-02
     3:  3.5784e-03 -2.0116e-01  2e-01  5e-03  6e-03
     4:  3.1010e-03 -1.7983e-02  2e-02  5e-04  6e-04
     5:  1.4043e-03 -1.2830e-02  1e-02  3e-04  3e-04
     6:  5.2299e-04 -3.6364e-03  4e-03  3e-05  3e-05
     7:  5.0208e-05 -7.0228e-04  8e-04  4e-17  4e-16
     8:  3.3647e-07 -4.7093e-05  5e-05  4e-17  6e-17
     9:  4.1459e-11 -5.1151e-07  5e-07  6e-17  6e-18
    10:  4.1461e-15 -5.1148e-09  5e-09  7e-17  6e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1464e-03 -1.0047e+00  6e+01  8e+00  1e+01
     1:  3.5679e-03 -9.6987e-01  2e+00  1e-01  2e-01
     2:  3.6669e-03 -5.8085e-01  8e-01  5e-02  6e-02
     3:  3.6073e-03 -2.4508e-01  3e-01  7e-03  9e-03
     4:  3.2700e-03 -1.8550e-02  2e-02  5e-04  7e-04
     5:  1.3387e-03 -1.2339e-02  1e-02  3e-04  3e-04
     6:  5.0218e-04 -3.4850e-03  4e-03  2e-05  3e-05
     7:  4.9011e-05 -7.1598e-04  8e-04  1e-16  5e-16
     8:  3.4190e-07 -4.6329e-05  5e-05  7e-17  1e-16
     9:  4.0233e-11 -4.9433e-07  5e-07  1e-16  7e-18
    10:  4.0234e-15 -4.9430e-09  5e-09  4e-16  7e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1456e-03 -1.0036e+00  6e+01  8e+00  1e+01
     1:  3.5648e-03 -9.6886e-01  2e+00  1e-01  2e-01
     2:  3.6620e-03 -6.0731e-01  9e-01  6e-02  7e-02
     3:  3.6350e-03 -2.8489e-01  3e-01  9e-03  1e-02
     4:  3.4231e-03 -1.8564e-02  2e-02  5e-04  6e-04
     5:  1.1638e-03 -1.1155e-02  1e-02  2e-04  2e-04
     6:  4.0696e-04 -2.8308e-03  3e-03  3e-05  3e-05
     7:  3.5046e-05 -5.9317e-04  6e-04  5e-17  3e-16
     8:  1.8107e-07 -3.3399e-05  3e-05  5e-17  8e-17
     9:  1.9904e-11 -3.4658e-07  3e-07  7e-17  7e-18
    10:  1.9903e-15 -3.4656e-09  3e-09  4e-17  7e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1448e-03 -1.0026e+00  6e+01  8e+00  1e+01
     1:  3.5616e-03 -9.6783e-01  2e+00  2e-01  2e-01
     2:  3.6566e-03 -6.3028e-01  1e+00  6e-02  8e-02
     3:  3.6626e-03 -3.2041e-01  4e-01  1e-02  1e-02
     4:  3.5721e-03 -1.8215e-02  2e-02  4e-04  4e-04
     5:  9.5263e-04 -9.7849e-03  1e-02  9e-05  1e-04
     6:  3.1724e-04 -2.2597e-03  3e-03  2e-05  2e-05
     7:  2.2582e-05 -4.3735e-04  5e-04  5e-17  3e-16
     8:  6.7535e-08 -2.0152e-05  2e-05  2e-16  5e-17
     9:  7.0544e-12 -2.0474e-07  2e-07  4e-16  9e-18
    10:  7.0543e-16 -2.0474e-09  2e-09  1e-16  1e-17
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1440e-03 -1.0016e+00  6e+01  8e+00  1e+01
     1:  3.5584e-03 -9.6681e-01  2e+00  2e-01  2e-01
     2:  3.6507e-03 -6.5018e-01  1e+00  7e-02  9e-02
     3:  3.6910e-03 -3.5176e-01  4e-01  1e-02  2e-02
     4:  3.7236e-03 -1.7658e-02  2e-02  1e-04  1e-04
     5:  7.4050e-04 -8.5540e-03  9e-03  2e-05  2e-05
     6:  2.3912e-04 -1.7731e-03  2e-03  3e-06  4e-06
     7:  1.2866e-05 -2.9502e-04  3e-04  3e-16  2e-16
     8:  1.7693e-08 -9.9219e-06  1e-05  2e-16  4e-17
     9:  1.7913e-12 -9.9572e-08  1e-07  3e-16  6e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1432e-03 -1.0005e+00  6e+01  8e+00  1e+01
     1:  3.5553e-03 -9.6578e-01  2e+00  2e-01  2e-01
     2:  3.6443e-03 -6.6734e-01  1e+00  7e-02  9e-02
     3:  3.7208e-03 -3.7915e-01  5e-01  2e-02  2e-02
     4:  3.8787e-03 -2.2469e-02  3e-02  6e-17  5e-14
     5:  8.9561e-04 -4.5738e-03  5e-03  2e-16  5e-15
     6:  1.1259e-04 -1.2301e-03  1e-03  2e-16  2e-15
     7:  4.3752e-06 -1.5470e-04  2e-04  5e-17  3e-16
     8:  2.4127e-09 -3.5679e-06  4e-06  1e-16  1e-17
     9:  2.4195e-13 -3.5698e-08  4e-08  2e-16  9e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1425e-03 -9.9945e-01  6e+01  8e+00  1e+01
     1:  3.5521e-03 -9.6474e-01  2e+00  2e-01  2e-01
     2:  3.6375e-03 -6.8206e-01  1e+00  8e-02  1e-01
     3:  3.7528e-03 -4.0286e-01  5e-01  2e-02  2e-02
     4:  4.0351e-03 -3.1194e-02  4e-02  2e-17  3e-14
     5:  1.7728e-03 -7.4171e-03  9e-03  2e-16  7e-15
     6:  3.0314e-04 -3.0787e-03  3e-03  2e-16  7e-15
     7:  5.3891e-05 -5.4651e-04  6e-04  6e-17  6e-16
     8:  1.0335e-06 -9.2090e-05  9e-05  8e-17  1e-16
     9:  2.6146e-10 -1.2578e-06  1e-06  2e-16  1e-17
    10:  2.6154e-14 -1.2577e-08  1e-08  2e-16  7e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1417e-03 -9.9839e-01  6e+01  8e+00  1e+01
     1:  3.5490e-03 -9.6369e-01  2e+00  2e-01  2e-01
     2:  3.6303e-03 -6.9457e-01  1e+00  9e-02  1e-01
     3:  3.7874e-03 -4.2318e-01  5e-01  2e-02  3e-02
     4:  4.1940e-03 -4.0450e-02  4e-02  2e-16  2e-14
     5:  2.2436e-03 -8.7036e-03  1e-02  6e-17  8e-15
     6:  3.8934e-04 -4.7089e-03  5e-03  2e-16  7e-15
     7:  1.1565e-04 -9.6668e-04  1e-03  7e-17  1e-15
     8:  4.8146e-06 -1.9308e-04  2e-04  7e-17  3e-16
     9:  8.1916e-09 -5.9040e-06  6e-06  6e-17  3e-17
    10:  8.3132e-13 -5.9349e-08  6e-08  2e-16  6e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1409e-03 -9.9733e-01  6e+01  8e+00  1e+01
     1:  3.5458e-03 -9.6264e-01  2e+00  2e-01  2e-01
     2:  3.6228e-03 -7.0507e-01  1e+00  9e-02  1e-01
     3:  3.8254e-03 -4.4039e-01  6e-01  2e-02  3e-02
     4:  4.3559e-03 -4.9909e-02  5e-02  2e-16  3e-14
     5:  2.6104e-03 -9.5780e-03  1e-02  6e-17  7e-15
     6:  5.7254e-04 -6.0392e-03  7e-03  1e-16  1e-14
     7:  1.7561e-04 -1.2170e-03  1e-03  2e-16  2e-15
     8:  1.4154e-05 -3.2718e-04  3e-04  1e-16  4e-16
     9:  9.8266e-08 -1.8736e-05  2e-05  6e-17  1e-16
    10:  1.0993e-11 -1.9557e-07  2e-07  1e-16  8e-18
    11:  1.0995e-15 -1.9556e-09  2e-09  2e-16  7e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1401e-03 -9.9628e-01  6e+01  8e+00  1e+01
     1:  3.5427e-03 -9.6158e-01  2e+00  2e-01  2e-01
     2:  3.6149e-03 -7.1373e-01  1e+00  1e-01  1e-01
     3:  3.8674e-03 -4.5473e-01  6e-01  2e-02  3e-02
     4:  4.5208e-03 -5.9263e-02  6e-02  2e-16  3e-14
     5:  2.9287e-03 -1.0305e-02  1e-02  6e-17  7e-15
     6:  4.2200e-04 -5.9361e-03  6e-03  1e-16  2e-14
     7:  1.5682e-04 -1.2355e-03  1e-03  1e-16  3e-15
     8:  1.5483e-05 -3.2636e-04  3e-04  1e-16  6e-16
     9:  1.2938e-07 -2.0431e-05  2e-05  2e-16  8e-17
    10:  1.5609e-11 -2.1990e-07  2e-07  2e-16  1e-17
    11:  1.5610e-15 -2.1989e-09  2e-09  1e-16  9e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1393e-03 -9.9522e-01  6e+01  8e+00  1e+01
     1:  3.5395e-03 -9.6052e-01  2e+00  2e-01  3e-01
     2:  3.6067e-03 -7.2068e-01  1e+00  1e-01  1e-01
     3:  3.9142e-03 -4.6646e-01  6e-01  3e-02  3e-02
     4:  4.6887e-03 -6.8230e-02  7e-02  6e-17  4e-14
     5:  3.2159e-03 -1.0869e-02  1e-02  2e-16  9e-15
     6:  4.3352e-04 -5.9117e-03  6e-03  1e-16  1e-14
     7:  1.5950e-04 -1.1338e-03  1e-03  2e-16  2e-15
     8:  1.3079e-05 -3.0754e-04  3e-04  6e-17  5e-16
     9:  1.2002e-07 -1.7453e-05  2e-05  1e-16  9e-17
    10:  1.5088e-11 -1.9222e-07  2e-07  7e-17  9e-18
    11:  1.5087e-15 -1.9221e-09  2e-09  2e-17  8e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1385e-03 -9.9415e-01  6e+01  8e+00  1e+01
     1:  3.5364e-03 -9.5945e-01  3e+00  2e-01  3e-01
     2:  3.5981e-03 -7.2604e-01  1e+00  1e-01  1e-01
     3:  3.9668e-03 -4.7578e-01  7e-01  3e-02  4e-02
     4:  4.8593e-03 -7.6543e-02  8e-02  1e-16  8e-14
     5:  3.4802e-03 -1.1308e-02  1e-02  7e-17  2e-14
     6:  4.7413e-04 -6.0785e-03  7e-03  2e-16  1e-14
     7:  1.6882e-04 -1.0554e-03  1e-03  1e-16  2e-15
     8:  1.8084e-05 -3.7440e-04  4e-04  1e-16  1e-15
     9:  3.2968e-07 -2.7510e-05  3e-05  1e-16  9e-17
    10:  5.9385e-11 -3.5245e-07  4e-07  1e-16  1e-17
    11:  5.9394e-15 -3.5241e-09  4e-09  8e-17  7e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1377e-03 -9.9309e-01  6e+01  8e+00  1e+01
     1:  3.5332e-03 -9.5838e-01  3e+00  2e-01  3e-01
     2:  3.5893e-03 -7.2989e-01  1e+00  1e-01  1e-01
     3:  4.0263e-03 -4.8290e-01  7e-01  3e-02  4e-02
     4:  5.0322e-03 -8.3949e-02  9e-02  4e-16  1e-13
     5:  3.7263e-03 -1.1638e-02  2e-02  2e-17  2e-14
     6:  5.4753e-04 -6.5950e-03  7e-03  2e-16  9e-15
     7:  1.9311e-04 -1.1525e-03  1e-03  6e-17  2e-15
     8:  1.9387e-05 -3.2471e-04  3e-04  2e-16  1e-15
     9:  4.5477e-07 -2.9436e-05  3e-05  1e-16  2e-16
    10:  9.6560e-11 -4.0380e-07  4e-07  5e-17  1e-17
    11:  9.6584e-15 -4.0375e-09  4e-09  6e-17  8e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1369e-03 -9.9202e-01  6e+01  8e+00  1e+01
     1:  3.5301e-03 -9.5730e-01  3e+00  2e-01  3e-01
     2:  3.5801e-03 -7.3230e-01  2e+00  1e-01  1e-01
     3:  4.0944e-03 -4.8796e-01  7e-01  3e-02  4e-02
     4:  5.2071e-03 -9.0196e-02  1e-01  6e-17  9e-14
     5:  3.9597e-03 -1.2087e-02  2e-02  2e-16  1e-14
     6:  6.2329e-04 -6.8818e-03  8e-03  4e-16  1e-14
     7:  2.0990e-04 -1.1083e-03  1e-03  3e-16  2e-15
     8:  2.3216e-05 -3.2116e-04  3e-04  7e-17  1e-15
     9:  7.1298e-07 -3.2928e-05  3e-05  8e-17  4e-16
    10:  2.1477e-10 -5.2729e-07  5e-07  7e-17  1e-17
    11:  2.1489e-14 -5.2721e-09  5e-09  8e-17  9e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1362e-03 -9.9095e-01  6e+01  8e+00  1e+01
     1:  3.5270e-03 -9.5621e-01  3e+00  2e-01  3e-01
     2:  3.5708e-03 -7.3332e-01  2e+00  1e-01  2e-01
     3:  4.1729e-03 -4.9114e-01  7e-01  3e-02  4e-02
     4:  5.3834e-03 -9.5026e-02  1e-01  1e-16  6e-14
     5:  4.1770e-03 -1.2391e-02  2e-02  9e-17  1e-14
     6:  7.0465e-04 -7.0781e-03  8e-03  2e-16  2e-14
     7:  2.4004e-04 -1.2517e-03  1e-03  4e-16  3e-15
     8:  4.0714e-05 -3.2364e-04  4e-04  4e-16  2e-15
     9:  1.0901e-06 -3.7271e-05  4e-05  2e-16  1e-15
    10:  6.5204e-10 -7.9824e-07  8e-07  2e-16  3e-17
    11:  6.5338e-14 -7.9833e-09  8e-09  1e-16  7e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1354e-03 -9.8988e-01  6e+01  8e+00  1e+01
     1:  3.5238e-03 -9.5512e-01  3e+00  2e-01  3e-01
     2:  3.5611e-03 -7.3299e-01  2e+00  1e-01  2e-01
     3:  4.2644e-03 -4.9254e-01  8e-01  4e-02  4e-02
     4:  5.5602e-03 -9.8159e-02  1e-01  1e-16  1e-13
     5:  4.3782e-03 -1.2550e-02  2e-02  3e-16  2e-14
     6:  7.8825e-04 -7.2034e-03  8e-03  2e-16  2e-14
     7:  2.7532e-04 -1.3664e-03  2e-03  4e-17  6e-15
     8:  6.2505e-05 -3.4018e-04  4e-04  2e-16  1e-15
     9:  4.4285e-06 -8.0981e-05  9e-05  6e-17  1e-15
    10:  1.7809e-08 -3.6599e-06  4e-06  7e-17  1e-16
    11:  1.9057e-12 -3.7581e-08  4e-08  2e-16  9e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1346e-03 -9.8881e-01  6e+01  8e+00  1e+01
     1:  3.5207e-03 -9.5402e-01  3e+00  2e-01  3e-01
     2:  3.5512e-03 -7.3133e-01  2e+00  1e-01  2e-01
     3:  4.3725e-03 -4.9227e-01  8e-01  4e-02  4e-02
     4:  5.7368e-03 -9.9272e-02  1e-01  2e-16  7e-14
     5:  4.5614e-03 -1.2559e-02  2e-02  6e-17  2e-14
     6:  8.7343e-04 -7.2649e-03  8e-03  4e-16  3e-14
     7:  3.1549e-04 -1.4411e-03  2e-03  4e-16  6e-15
     8:  9.2042e-05 -4.1107e-04  5e-04  2e-16  3e-15
     9:  1.1886e-05 -1.3215e-04  1e-04  1e-16  2e-15
    10:  2.0434e-07 -1.0822e-05  1e-05  6e-17  9e-16
    11:  5.6305e-11 -1.7182e-07  2e-07  2e-16  2e-17
    12:  5.6327e-15 -1.7179e-09  2e-09  2e-16  1e-17
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1338e-03 -9.8774e-01  6e+01  8e+00  1e+01
     1:  3.5176e-03 -9.5292e-01  3e+00  3e-01  3e-01
     2:  3.5411e-03 -7.2835e-01  2e+00  1e-01  2e-01
     3:  4.5023e-03 -4.9040e-01  8e-01  4e-02  4e-02
     4:  5.9118e-03 -9.7969e-02  1e-01  1e-16  8e-14
     5:  4.7217e-03 -1.2407e-02  2e-02  2e-16  2e-14
     6:  9.6053e-04 -7.2608e-03  8e-03  8e-17  3e-14
     7:  3.6025e-04 -1.4791e-03  2e-03  2e-16  6e-15
     8:  1.1236e-04 -4.1701e-04  5e-04  3e-16  2e-15
     9:  1.9216e-05 -1.5476e-04  2e-04  4e-16  4e-15
    10:  7.6266e-07 -1.8023e-05  2e-05  6e-16  1e-15
    11:  8.3746e-10 -5.4390e-07  5e-07  2e-16  1e-16
    12:  8.4477e-14 -5.4530e-09  5e-09  3e-16  6e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1330e-03 -9.8666e-01  6e+01  8e+00  1e+01
     1:  3.5145e-03 -9.5181e-01  3e+00  3e-01  3e-01
     2:  3.5308e-03 -7.2402e-01  2e+00  1e-01  2e-01
     3:  4.6612e-03 -4.8699e-01  8e-01  4e-02  4e-02
     4:  6.0833e-03 -9.3728e-02  1e-01  1e-16  1e-13
     5:  4.8495e-03 -1.2063e-02  2e-02  2e-16  2e-14
     6:  1.0515e-03 -7.1781e-03  8e-03  6e-17  3e-14
     7:  4.0886e-04 -1.4783e-03  2e-03  2e-16  5e-15
     8:  1.3094e-04 -3.9489e-04  5e-04  8e-17  3e-15
     9:  2.9955e-05 -1.6745e-04  2e-04  9e-17  5e-15
    10:  2.1171e-06 -2.3757e-05  3e-05  1e-16  1e-15
    11:  9.7475e-09 -1.4471e-06  1e-06  1e-16  2e-16
    12:  1.1024e-12 -1.5213e-08  2e-08  4e-17  9e-18
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1323e-03 -9.8558e-01  6e+01  8e+00  1e+01
     1:  3.5113e-03 -9.5069e-01  3e+00  3e-01  3e-01
     2:  3.5203e-03 -7.1832e-01  2e+00  1e-01  2e-01
     3:  4.8604e-03 -4.8203e-01  8e-01  3e-02  4e-02
     4:  6.2483e-03 -8.5815e-02  9e-02  2e-16  8e-14
     5:  4.9246e-03 -1.1465e-02  2e-02  2e-16  2e-14
     6:  1.3283e-03 -7.2153e-03  9e-03  2e-16  3e-14
     7:  5.6604e-04 -1.8126e-03  2e-03  2e-16  7e-15
     8:  1.6880e-04 -3.8277e-04  6e-04  2e-16  5e-15
     9:  4.5891e-05 -1.7419e-04  2e-04  2e-16  3e-15
    10:  5.2035e-06 -2.5819e-05  3e-05  1e-16  2e-15
    11:  1.3883e-07 -3.4760e-06  4e-06  2e-16  1e-15
    12:  8.1375e-11 -7.6795e-08  8e-08  2e-16  5e-17
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1315e-03 -9.8450e-01  6e+01  8e+00  1e+01
     1:  3.5082e-03 -9.4957e-01  3e+00  3e-01  3e-01
     2:  3.5097e-03 -7.1122e-01  2e+00  1e-01  2e-01
     3:  5.1182e-03 -4.7545e-01  7e-01  3e-02  4e-02
     4:  6.4017e-03 -7.3122e-02  8e-02  6e-16  8e-14
     5:  4.9025e-03 -1.0648e-02  2e-02  2e-16  3e-14
     6:  1.7483e-03 -7.0275e-03  9e-03  2e-16  2e-14
     7:  7.4934e-04 -1.8611e-03  3e-03  1e-16  8e-15
     8:  2.2285e-04 -4.3139e-04  7e-04  2e-16  5e-15
     9:  4.6954e-05 -1.1674e-04  2e-04  2e-16  6e-15
    10:  8.4142e-06 -1.8319e-05  3e-05  6e-16  3e-15
    11:  7.7676e-07 -3.4488e-06  4e-06  4e-16  4e-15
    12:  1.8108e-08 -3.8723e-07  4e-07  3e-16  3e-15
    13:  7.5991e-12 -7.4552e-09  7e-09  2e-16  6e-17
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1307e-03 -9.8342e-01  6e+01  8e+00  1e+01
     1:  3.5051e-03 -9.4845e-01  3e+00  3e-01  4e-01
     2:  3.4990e-03 -7.0264e-01  2e+00  1e-01  2e-01
     3:  5.4654e-03 -4.6711e-01  7e-01  3e-02  3e-02
     4:  6.5333e-03 -5.3922e-02  6e-02  6e-17  7e-14
     5:  4.6554e-03 -9.6521e-03  1e-02  1e-16  2e-14
     6:  1.3206e-03 -5.0433e-03  6e-03  2e-16  2e-14
     7:  4.7291e-04 -8.2911e-04  1e-03  2e-16  7e-15
     8:  1.8816e-04 -2.2244e-04  4e-04  4e-16  5e-15
     9:  6.1265e-05 -7.9170e-05  1e-04  4e-16  1e-14
    10:  1.5087e-05 -5.6943e-06  2e-05  2e-16  3e-15
    11:  4.1729e-06  1.3324e-06  3e-06  1e-16  1e-14
    12:  2.1386e-06  1.8525e-06  3e-07  2e-16  7e-15
    13:  1.9116e-06  1.8758e-06  4e-08  1e-16  2e-15
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1299e-03 -9.8234e-01  6e+01  8e+00  1e+01
     1:  3.5020e-03 -9.4731e-01  3e+00  3e-01  4e-01
     2:  3.4883e-03 -6.9252e-01  2e+00  1e-01  2e-01
     3:  5.9602e-03 -4.5663e-01  6e-01  2e-02  2e-02
     4:  6.6192e-03 -2.5975e-02  3e-02  1e-16  4e-14
     5:  3.2835e-03 -6.5225e-03  1e-02  9e-17  2e-14
     6:  9.9748e-04 -1.5772e-03  3e-03  2e-16  2e-14
     7:  3.5569e-04 -5.4155e-04  9e-04  2e-16  6e-15
     8:  1.5523e-04 -1.0435e-04  3e-04  2e-16  5e-15
     9:  6.8935e-05 -1.9373e-05  9e-05  2e-16  1e-14
    10:  2.8984e-05  1.4123e-05  1e-05  1e-16  1e-14
    11:  1.8254e-05  1.7071e-05  1e-06  1e-16  3e-14
    12:  1.7287e-05  1.7237e-05  5e-08  2e-16  6e-15
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1291e-03 -9.8125e-01  6e+01  8e+00  1e+01
     1:  3.4989e-03 -9.4617e-01  3e+00  3e-01  4e-01
     2:  3.4776e-03 -6.8073e-01  2e+00  2e-01  2e-01
     3:  6.7237e-03 -4.4327e-01  5e-01  3e-03  4e-03
     4:  6.5970e-03 -1.5921e-02  2e-02  2e-04  2e-04
     5:  1.9023e-03 -5.9529e-03  8e-03  2e-16  2e-14
     6:  8.8653e-04 -1.4653e-03  2e-03  1e-16  5e-15
     7:  3.3158e-04 -2.1544e-04  5e-04  2e-16  6e-15
     8:  1.2807e-04 -3.4658e-05  2e-04  1e-16  7e-15
     9:  6.7080e-05  4.4190e-05  2e-05  1e-16  1e-14
    10:  5.1621e-05  4.7131e-05  4e-06  2e-16  8e-14
    11:  4.9307e-05  4.7893e-05  1e-06  2e-16  4e-14
    12:  4.8122e-05  4.8078e-05  4e-08  6e-17  2e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1284e-03 -9.8017e-01  6e+01  8e+00  1e+01
     1:  3.4958e-03 -9.4503e-01  3e+00  3e-01  4e-01
     2:  3.4670e-03 -6.6715e-01  2e+00  2e-01  2e-01
     3:  7.3709e-03 -4.6725e-01  5e-01  2e-16  5e-12
     4:  7.0484e-03 -1.6444e-02  2e-02  1e-16  2e-13
     5:  2.1492e-03 -6.0055e-03  8e-03  2e-16  3e-14
     6:  1.0319e-03 -1.4361e-03  2e-03  3e-16  1e-14
     7:  4.2590e-04 -2.5386e-04  7e-04  4e-16  6e-15
     8:  2.2896e-04  2.2556e-05  2e-04  7e-17  7e-15
     9:  1.4190e-04  7.7002e-05  6e-05  1e-16  2e-14
    10:  9.9017e-05  9.3597e-05  5e-06  5e-17  4e-14
    11:  9.4695e-05  9.4341e-05  4e-07  2e-16  6e-14
    12:  9.4407e-05  9.4402e-05  5e-09  1e-16  2e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1276e-03 -9.7908e-01  6e+01  8e+00  1e+01
     1:  3.4927e-03 -9.4388e-01  3e+00  3e-01  4e-01
     2:  3.4567e-03 -6.5163e-01  2e+00  2e-01  2e-01
     3:  8.0361e-03 -5.0602e-01  5e-01  2e-16  3e-12
     4:  7.7363e-03 -1.7626e-02  3e-02  1e-16  2e-13
     5:  2.4902e-03 -7.4843e-03  1e-02  2e-16  1e-14
     6:  1.2938e-03 -1.4891e-03  3e-03  1e-16  7e-15
     7:  5.5936e-04 -3.6190e-04  9e-04  2e-16  7e-15
     8:  3.2651e-04  5.7514e-05  3e-04  1e-16  1e-14
     9:  2.3504e-04  1.1545e-04  1e-04  8e-17  3e-14
    10:  1.6735e-04  1.5408e-04  1e-05  8e-17  4e-14
    11:  1.5708e-04  1.5594e-04  1e-06  8e-17  2e-13
    12:  1.5622e-04  1.5620e-04  2e-08  2e-16  1e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1268e-03 -9.7799e-01  6e+01  8e+00  1e+01
     1:  3.4896e-03 -9.4272e-01  3e+00  3e-01  4e-01
     2:  3.4469e-03 -6.3395e-01  2e+00  2e-01  2e-01
     3:  8.8447e-03 -5.5184e-01  6e-01  1e-16  4e-12
     4:  8.5708e-03 -1.9218e-02  3e-02  2e-16  2e-13
     5:  2.8400e-03 -9.5958e-03  1e-02  2e-16  2e-14
     6:  1.6351e-03 -1.7316e-03  3e-03  6e-17  1e-14
     7:  6.8527e-04 -3.9994e-04  1e-03  3e-16  2e-14
     8:  4.0032e-04  1.4530e-04  3e-04  2e-16  1e-14
     9:  3.1282e-04  1.9237e-04  1e-04  1e-16  4e-14
    10:  2.4352e-04  2.3160e-04  1e-05  1e-16  1e-13
    11:  2.3398e-04  2.3334e-04  6e-07  2e-16  1e-13
    12:  2.3348e-04  2.3347e-04  7e-09  8e-17  8e-14
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1260e-03 -9.7689e-01  6e+01  8e+00  1e+01
     1:  3.4865e-03 -9.4155e-01  3e+00  3e-01  4e-01
     2:  3.4377e-03 -6.1389e-01  2e+00  2e-01  2e-01
     3:  9.8343e-03 -6.0580e-01  6e-01  1e-16  4e-12
     4:  9.5928e-03 -2.1307e-02  3e-02  2e-16  2e-13
     5:  3.2070e-03 -1.2546e-02  2e-02  9e-17  2e-14
     6:  2.0423e-03 -2.1458e-03  4e-03  3e-17  3e-14
     7:  1.1402e-03 -1.1292e-03  2e-03  2e-16  2e-14
     8:  6.2051e-04  9.2538e-05  5e-04  7e-17  2e-14
     9:  4.2900e-04  2.9540e-04  1e-04  1e-16  1e-14
    10:  3.5861e-04  3.1505e-04  4e-05  3e-16  1e-13
    11:  3.2733e-04  3.2599e-04  1e-06  1e-16  3e-14
    12:  3.2624e-04  3.2622e-04  1e-08  2e-16  7e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1253e-03 -9.7580e-01  6e+01  8e+00  1e+01
     1:  3.4834e-03 -9.4039e-01  3e+00  3e-01  4e-01
     2:  3.4296e-03 -5.9114e-01  2e+00  2e-01  2e-01
     3:  1.0726e-02 -6.6624e-01  7e-01  5e-03  6e-03
     4:  1.0551e-02 -2.3074e-02  3e-02  2e-04  3e-04
     5:  3.7200e-03 -1.4646e-02  2e-02  2e-16  2e-14
     6:  2.4407e-03 -2.2624e-03  5e-03  4e-16  8e-14
     7:  1.4934e-03 -1.5443e-03  3e-03  2e-16  5e-14
     8:  7.8598e-04  1.7667e-04  6e-04  2e-16  3e-14
     9:  6.1311e-04  3.4429e-04  3e-04  5e-17  2e-14
    10:  4.9401e-04  4.2158e-04  7e-05  1e-16  8e-14
    11:  4.4683e-04  4.3034e-04  2e-05  4e-16  2e-13
    12:  4.3516e-04  4.3430e-04  9e-07  3e-16  3e-14
    13:  4.3446e-04  4.3445e-04  1e-08  6e-17  1e-12
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1245e-03 -9.7470e-01  6e+01  8e+00  1e+01
     1:  3.4804e-03 -9.3921e-01  4e+00  3e-01  4e-01
     2:  3.4228e-03 -5.6532e-01  2e+00  2e-01  2e-01
     3:  1.1160e-02 -7.2105e-01  9e-01  2e-02  2e-02
     4:  1.1038e-02 -2.4452e-02  4e-02  2e-04  2e-04
     5:  5.7331e-03 -7.4500e-03  1e-02  4e-16  3e-14
     6:  2.4552e-03 -3.8096e-03  6e-03  2e-16  1e-13
     7:  1.6004e-03 -7.2398e-04  2e-03  2e-16  4e-14
     8:  9.4302e-04  2.7376e-04  7e-04  2e-16  2e-14
     9:  6.2031e-04  5.4583e-04  7e-05  1e-16  8e-14
    10:  5.6300e-04  5.5756e-04  5e-06  2e-16  7e-14
    11:  5.5872e-04  5.5807e-04  7e-07  2e-16  2e-13
    12:  5.5817e-04  5.5816e-04  7e-09  1e-16  4e-14
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1237e-03 -9.7360e-01  6e+01  8e+00  1e+01
     1:  3.4773e-03 -9.3803e-01  4e+00  3e-01  4e-01
     2:  3.4181e-03 -5.3600e-01  2e+00  1e-01  2e-01
     3:  1.1545e-02 -7.6587e-01  1e+00  3e-02  3e-02
     4:  1.1477e-02 -5.5358e-02  7e-02  5e-04  6e-04
     5:  1.0505e-02 -1.3080e-02  2e-02  2e-04  2e-04
     6:  6.0533e-03 -2.1304e-02  3e-02  1e-04  2e-04
     7:  5.2491e-03 -7.4075e-03  1e-02  5e-05  6e-05
     8:  4.4884e-03 -9.9891e-03  1e-02  4e-05  5e-05
     9:  2.3500e-03 -2.8338e-03  5e-03  2e-16  4e-14
    10:  1.5106e-03 -1.6301e-04  2e-03  2e-16  3e-14
    11:  1.0949e-03  3.7126e-04  7e-04  2e-16  6e-14
    12:  8.0121e-04  6.6521e-04  1e-04  7e-17  4e-14
    13:  7.1768e-04  6.9230e-04  3e-05  2e-16  2e-13
    14:  6.9908e-04  6.9865e-04  4e-07  1e-16  2e-13
    15:  6.9873e-04  6.9872e-04  4e-09  1e-16  3e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1230e-03 -9.7250e-01  6e+01  8e+00  1e+01
     1:  3.4742e-03 -9.3684e-01  4e+00  4e-01  4e-01
     2:  3.4162e-03 -5.0260e-01  2e+00  1e-01  2e-01
     3:  1.1313e-02 -7.8124e-01  1e+00  4e-02  5e-02
     4:  1.1479e-02 -1.0572e-01  1e-01  1e-03  1e-03
     5:  1.1004e-02 -1.2998e-02  2e-02  2e-04  3e-04
     6:  5.1487e-03 -2.3775e-02  3e-02  2e-04  2e-04
     7:  4.3660e-03 -4.4570e-03  9e-03  4e-05  5e-05
     8:  3.9023e-03 -4.5436e-03  8e-03  4e-05  5e-05
     9:  1.9956e-03 -1.8116e-03  4e-03  8e-17  5e-14
    10:  1.3507e-03  3.8655e-04  1e-03  2e-16  2e-14
    11:  1.0873e-03  7.5350e-04  3e-04  2e-16  5e-14
    12:  9.4947e-04  8.4656e-04  1e-04  4e-16  9e-14
    13:  8.9036e-04  8.5466e-04  4e-05  2e-16  5e-13
    14:  8.6373e-04  8.6302e-04  7e-07  1e-16  6e-14
    15:  8.6318e-04  8.6317e-04  7e-09  7e-17  1e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1222e-03 -9.7140e-01  6e+01  8e+00  1e+01
     1:  3.4711e-03 -9.3565e-01  4e+00  4e-01  4e-01
     2:  3.4182e-03 -4.6440e-01  2e+00  1e-01  2e-01
     3:  1.0199e-02 -7.4467e-01  1e+00  6e-02  7e-02
     4:  1.1089e-02 -1.5830e-01  2e-01  2e-03  2e-03
     5:  1.0756e-02 -1.2275e-02  2e-02  3e-04  3e-04
     6:  4.2059e-03 -2.5627e-02  3e-02  9e-05  1e-04
     7:  3.6120e-03 -4.6619e-03  8e-03  2e-05  3e-05
     8:  2.8198e-03 -4.4112e-03  7e-03  1e-05  2e-05
     9:  1.7137e-03  4.9541e-04  1e-03  1e-06  2e-06
    10:  1.4624e-03  7.6955e-04  7e-04  4e-07  5e-07
    11:  1.2125e-03  1.0015e-03  2e-04  1e-07  2e-07
    12:  1.1609e-03  1.0017e-03  2e-04  7e-08  8e-08
    13:  1.0626e-03  1.0482e-03  1e-05  6e-09  7e-09
    14:  1.0526e-03  1.0525e-03  1e-07  6e-11  7e-11
    15:  1.0525e-03  1.0525e-03  1e-09  6e-13  8e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1214e-03 -9.7029e-01  6e+01  8e+00  1e+01
     1:  3.4681e-03 -9.3445e-01  4e+00  4e-01  5e-01
     2:  3.4257e-03 -4.2048e-01  2e+00  1e-01  2e-01
     3:  9.5459e-03 -6.7945e-01  1e+00  6e-02  7e-02
     4:  1.1055e-02 -1.5350e-01  2e-01  2e-03  2e-03
     5:  1.0719e-02 -1.1160e-02  2e-02  2e-04  3e-04
     6:  4.2098e-03 -2.3690e-02  3e-02  7e-05  8e-05
     7:  3.6652e-03 -3.9572e-03  8e-03  2e-05  2e-05
     8:  2.9314e-03 -3.6387e-03  7e-03  1e-05  1e-05
     9:  1.8920e-03  7.6972e-04  1e-03  1e-06  1e-06
    10:  1.7890e-03  8.6311e-04  9e-04  5e-07  7e-07
    11:  1.5462e-03  1.1425e-03  4e-04  8e-08  1e-07
    12:  1.3277e-03  1.2540e-03  7e-05  1e-08  2e-08
    13:  1.2731e-03  1.2647e-03  8e-06  1e-09  1e-09
    14:  1.2668e-03  1.2667e-03  9e-08  1e-11  1e-11
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1207e-03 -9.6919e-01  6e+01  8e+00  1e+01
     1:  3.4650e-03 -9.3324e-01  4e+00  4e-01  5e-01
     2:  3.4361e-03 -3.7932e-01  2e+00  1e-01  2e-01
     3:  9.9337e-03 -5.9721e-01  1e+00  5e-02  6e-02
     4:  1.1318e-02 -1.1756e-01  1e-01  7e-04  8e-04
     5:  1.0871e-02 -9.8779e-03  2e-02  1e-04  1e-04
     6:  4.8476e-03 -2.0392e-02  3e-02  5e-05  7e-05
     7:  4.2096e-03 -2.7935e-03  7e-03  1e-05  2e-05
     8:  3.5724e-03 -2.6972e-03  6e-03  1e-05  1e-05
     9:  2.2109e-03  1.0217e-03  1e-03  3e-16  6e-14
    10:  1.8368e-03  1.3095e-03  5e-04  7e-17  2e-13
    11:  1.7193e-03  1.4169e-03  3e-04  1e-16  1e-13
    12:  1.5482e-03  1.4984e-03  5e-05  2e-16  3e-13
    13:  1.5079e-03  1.5052e-03  3e-06  7e-17  9e-13
    14:  1.5058e-03  1.5058e-03  3e-08  2e-16  2e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1199e-03 -9.6808e-01  6e+01  8e+00  1e+01
     1:  3.4619e-03 -9.3203e-01  4e+00  4e-01  5e-01
     2:  3.4357e-03 -3.6337e-01  2e+00  1e-01  2e-01
     3:  1.0290e-02 -5.8317e-01  1e+00  5e-02  6e-02
     4:  1.1657e-02 -1.2141e-01  1e-01  8e-04  1e-03
     5:  1.1253e-02 -9.1430e-03  2e-02  1e-04  1e-04
     6:  5.0151e-03 -1.9688e-02  2e-02  6e-05  8e-05
     7:  4.4192e-03 -2.1477e-03  7e-03  2e-05  2e-05
     8:  3.7920e-03 -2.1997e-03  6e-03  1e-05  1e-05
     9:  2.5022e-03  1.1966e-03  1e-03  7e-07  8e-07
    10:  2.4084e-03  1.2258e-03  1e-03  4e-07  4e-07
    11:  2.1155e-03  1.6010e-03  5e-04  1e-08  2e-08
    12:  1.8216e-03  1.7611e-03  6e-05  1e-09  2e-09
    13:  1.7720e-03  1.7692e-03  3e-06  5e-11  6e-11
    14:  1.7697e-03  1.7697e-03  3e-08  5e-13  9e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1191e-03 -9.6697e-01  6e+01  8e+00  1e+01
     1:  3.4589e-03 -9.3082e-01  4e+00  4e-01  5e-01
     2:  3.4364e-03 -3.4678e-01  2e+00  1e-01  2e-01
     3:  1.0697e-02 -5.6611e-01  1e+00  5e-02  6e-02
     4:  1.2007e-02 -1.2215e-01  1e-01  8e-04  1e-03
     5:  1.1632e-02 -8.3347e-03  2e-02  1e-04  1e-04
     6:  5.2826e-03 -1.8770e-02  2e-02  6e-05  8e-05
     7:  4.6765e-03 -1.4388e-03  6e-03  2e-05  2e-05
     8:  4.0221e-03 -1.6076e-03  6e-03  1e-05  1e-05
     9:  2.8721e-03  1.3122e-03  2e-03  2e-06  2e-06
    10:  2.8256e-03  1.2263e-03  2e-03  1e-06  1e-06
    11:  2.5583e-03  1.7290e-03  8e-04  2e-16  3e-14
    12:  2.1260e-03  2.0460e-03  8e-05  1e-16  4e-14
    13:  2.0598e-03  2.0583e-03  1e-06  1e-16  8e-13
    14:  2.0586e-03  2.0585e-03  1e-08  1e-16  2e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1184e-03 -9.6586e-01  6e+01  8e+00  1e+01
     1:  3.4558e-03 -9.2959e-01  4e+00  4e-01  5e-01
     2:  3.4382e-03 -3.2959e-01  2e+00  1e-01  2e-01
     3:  1.1182e-02 -5.4529e-01  1e+00  5e-02  6e-02
     4:  1.2374e-02 -1.1895e-01  1e-01  7e-04  9e-04
     5:  1.2016e-02 -7.5235e-03  2e-02  1e-04  1e-04
     6:  5.5234e-03 -1.7996e-02  2e-02  6e-05  7e-05
     7:  4.9087e-03 -7.3322e-04  6e-03  1e-05  2e-05
     8:  4.1550e-03 -1.0713e-03  5e-03  8e-06  1e-05
     9:  3.1957e-03  1.5636e-03  2e-03  2e-06  3e-06
    10:  3.1694e-03  1.3028e-03  2e-03  1e-06  2e-06
    11:  3.0404e-03  1.8083e-03  1e-03  1e-16  4e-14
    12:  2.5175e-03  2.3076e-03  2e-04  4e-16  1e-13
    13:  2.3815e-03  2.3708e-03  1e-05  6e-16  2e-13
    14:  2.3723e-03  2.3722e-03  1e-07  1e-16  4e-13
    15:  2.3722e-03  2.3722e-03  1e-09  2e-16  3e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1176e-03 -9.6474e-01  6e+01  8e+00  1e+01
     1:  3.4527e-03 -9.2836e-01  4e+00  4e-01  5e-01
     2:  3.4411e-03 -3.1185e-01  2e+00  2e-01  2e-01
     3:  1.1792e-02 -5.1943e-01  1e+00  4e-02  5e-02
     4:  1.2773e-02 -1.0968e-01  1e-01  5e-04  6e-04
     5:  1.2417e-02 -6.6666e-03  2e-02  8e-05  1e-04
     6:  5.8438e-03 -1.7174e-02  2e-02  5e-05  6e-05
     7:  5.1858e-03  2.5498e-05  5e-03  1e-05  1e-05
     8:  4.2279e-03 -5.8406e-04  5e-03  5e-06  6e-06
     9:  3.3332e-03  2.1405e-03  1e-03  1e-06  1e-06
    10:  3.0931e-03  2.4722e-03  6e-04  9e-08  1e-07
    11:  2.8391e-03  2.6808e-03  2e-04  5e-09  6e-09
    12:  2.7249e-03  2.7078e-03  2e-05  1e-10  1e-10
    13:  2.7110e-03  2.7108e-03  2e-07  1e-12  1e-12
    14:  2.7108e-03  2.7108e-03  2e-09  1e-14  2e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1168e-03 -9.6363e-01  6e+01  8e+00  1e+01
     1:  3.4497e-03 -9.2713e-01  4e+00  4e-01  5e-01
     2:  3.4441e-03 -2.9503e-01  2e+00  2e-01  2e-01
     3:  1.2372e-02 -4.9611e-01  9e-01  4e-02  5e-02
     4:  1.3165e-02 -1.0115e-01  1e-01  4e-04  5e-04
     5:  1.2819e-02 -5.7696e-03  2e-02  6e-05  8e-05
     6:  6.1816e-03 -1.6362e-02  2e-02  4e-05  5e-05
     7:  5.4729e-03  7.9095e-04  5e-03  8e-06  9e-06
     8:  4.3638e-03  2.0630e-04  4e-03  3e-06  4e-06
     9:  3.4577e-03  2.8406e-03  6e-04  3e-07  4e-07
    10:  3.1607e-03  3.0536e-03  1e-04  3e-11  4e-11
    11:  3.0986e-03  3.0708e-03  3e-05  2e-12  2e-12
    12:  3.0745e-03  3.0742e-03  3e-07  2e-14  2e-13
    13:  3.0743e-03  3.0743e-03  3e-09  2e-16  2e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1161e-03 -9.6251e-01  6e+01  8e+00  1e+01
     1:  3.4466e-03 -9.2589e-01  4e+00  4e-01  5e-01
     2:  3.4459e-03 -2.8076e-01  2e+00  2e-01  2e-01
     3:  1.2569e-02 -4.8867e-01  1e+00  4e-02  5e-02
     4:  1.3452e-02 -1.1074e-01  1e-01  5e-04  6e-04
     5:  1.3160e-02 -4.9480e-03  2e-02  7e-05  8e-05
     6:  6.1607e-03 -1.6015e-02  2e-02  4e-05  4e-05
     7:  5.5382e-03  1.4772e-03  4e-03  6e-06  8e-06
     8:  4.6182e-03  1.1525e-03  3e-03  2e-06  3e-06
     9:  3.7960e-03  3.2368e-03  6e-04  3e-07  4e-07
    10:  3.5808e-03  3.4050e-03  2e-04  8e-08  9e-08
    11:  3.4882e-03  3.4571e-03  3e-05  5e-09  6e-09
    12:  3.4749e-03  3.4595e-03  2e-05  8e-10  9e-10
    13:  3.4627e-03  3.4625e-03  2e-07  9e-12  1e-11
    14:  3.4626e-03  3.4626e-03  2e-09  9e-14  8e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1153e-03 -9.6139e-01  6e+01  8e+00  1e+01
     1:  3.4436e-03 -9.2464e-01  4e+00  4e-01  5e-01
     2:  3.4483e-03 -2.6624e-01  2e+00  2e-01  2e-01
     3:  1.2769e-02 -4.8001e-01  1e+00  4e-02  5e-02
     4:  1.3740e-02 -1.1873e-01  1e-01  6e-04  7e-04
     5:  1.3494e-02 -4.0676e-03  2e-02  7e-05  9e-05
     6:  6.1668e-03 -1.5405e-02  2e-02  3e-05  4e-05
     7:  5.6270e-03  2.1773e-03  3e-03  5e-06  6e-06
     8:  4.7511e-03  2.2114e-03  3e-03  1e-06  1e-06
     9:  4.0740e-03  3.7644e-03  3e-04  1e-07  1e-07
    10:  3.9294e-03  3.8560e-03  7e-05  2e-08  2e-08
    11:  3.8789e-03  3.8755e-03  3e-06  3e-16  1e-12
    12:  3.8760e-03  3.8759e-03  7e-08  2e-16  2e-12
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1145e-03 -9.6027e-01  6e+01  8e+00  1e+01
     1:  3.4406e-03 -9.2339e-01  4e+00  4e-01  5e-01
     2:  3.4512e-03 -2.5150e-01  2e+00  2e-01  2e-01
     3:  1.2972e-02 -4.7026e-01  1e+00  5e-02  6e-02
     4:  1.4031e-02 -1.2503e-01  1e-01  7e-04  8e-04
     5:  1.3822e-02 -3.1305e-03  2e-02  8e-05  1e-04
     6:  6.1369e-03 -1.4786e-02  2e-02  3e-05  3e-05
     7:  5.7115e-03  2.8358e-03  3e-03  4e-06  4e-06
     8:  5.0038e-03  3.1189e-03  2e-03  2e-07  2e-07
     9:  4.4311e-03  4.2719e-03  2e-04  1e-08  1e-08
    10:  4.3406e-03  4.3136e-03  3e-05  2e-09  2e-09
    11:  4.3204e-03  4.3199e-03  5e-07  1e-11  2e-11
    12:  4.3200e-03  4.3200e-03  5e-09  1e-13  6e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1138e-03 -9.5914e-01  6e+01  8e+00  1e+01
     1:  3.4375e-03 -9.2213e-01  4e+00  4e-01  5e-01
     2:  3.4548e-03 -2.3655e-01  2e+00  2e-01  2e-01
     3:  1.3178e-02 -4.5953e-01  1e+00  5e-02  6e-02
     4:  1.4323e-02 -1.2957e-01  1e-01  8e-04  9e-04
     5:  1.4147e-02 -2.1379e-03  2e-02  9e-05  1e-04
     6:  6.3552e-03 -1.3153e-02  2e-02  2e-05  3e-05
     7:  5.9604e-03  3.5799e-03  2e-03  3e-06  3e-06
     8:  5.5029e-03  3.8319e-03  2e-03  9e-07  1e-06
     9:  4.8945e-03  4.7186e-03  2e-04  6e-08  8e-08
    10:  4.8002e-03  4.7947e-03  6e-06  1e-09  1e-09
    11:  4.7976e-03  4.7975e-03  1e-07  1e-11  1e-11
    12:  4.7975e-03  4.7975e-03  6e-09  1e-13  6e-11
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1130e-03 -9.5802e-01  6e+01  8e+00  1e+01
     1:  3.4345e-03 -9.2086e-01  4e+00  5e-01  5e-01
     2:  3.4590e-03 -2.2141e-01  2e+00  2e-01  2e-01
     3:  1.3386e-02 -4.4790e-01  1e+00  6e-02  7e-02
     4:  1.4617e-02 -1.3230e-01  2e-01  9e-04  1e-03
     5:  1.4470e-02 -1.0909e-03  2e-02  9e-05  1e-04
     6:  7.0247e-03 -1.0270e-02  2e-02  2e-05  3e-05
     7:  6.4889e-03  4.4407e-03  2e-03  3e-06  3e-06
     8:  6.0266e-03  4.5511e-03  1e-03  1e-06  1e-06
     9:  5.4069e-03  5.2613e-03  1e-04  7e-08  8e-08
    10:  5.3186e-03  5.3159e-03  3e-06  8e-10  9e-10
    11:  5.3170e-03  5.3170e-03  3e-08  8e-12  1e-11
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1123e-03 -9.5689e-01  6e+01  8e+00  1e+01
     1:  3.4315e-03 -9.1959e-01  4e+00  5e-01  6e-01
     2:  3.4637e-03 -2.0611e-01  2e+00  2e-01  3e-01
     3:  1.3597e-02 -4.3547e-01  1e+00  6e-02  7e-02
     4:  1.4913e-02 -1.3320e-01  2e-01  1e-03  1e-03
     5:  1.4792e-02  1.0001e-05  1e-02  1e-04  1e-04
     6:  7.7591e-03 -7.6061e-03  2e-02  2e-05  3e-05
     7:  7.0784e-03  5.0453e-03  2e-03  3e-06  4e-06
     8:  6.2590e-03  5.4641e-03  8e-04  2e-16  2e-12
     9:  5.9124e-03  5.8789e-03  3e-05  1e-16  2e-13
    10:  5.8893e-03  5.8890e-03  4e-07  1e-16  3e-13
    11:  5.8891e-03  5.8891e-03  4e-09  1e-16  4e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1115e-03 -9.5576e-01  6e+01  8e+00  1e+01
     1:  3.4284e-03 -9.1831e-01  4e+00  5e-01  6e-01
     2:  3.4690e-03 -1.9065e-01  2e+00  2e-01  3e-01
     3:  1.3810e-02 -4.2230e-01  1e+00  6e-02  8e-02
     4:  1.5211e-02 -1.3225e-01  2e-01  1e-03  1e-03
     5:  1.5113e-02  1.1641e-03  1e-02  1e-04  1e-04
     6:  8.5384e-03 -5.1157e-03  1e-02  2e-05  3e-05
     7:  7.7129e-03  5.7225e-03  2e-03  3e-06  4e-06
     8:  6.8061e-03  6.3722e-03  4e-04  2e-16  2e-12
     9:  6.5232e-03  6.5095e-03  1e-05  2e-16  3e-13
    10:  6.5138e-03  6.5137e-03  1e-07  2e-16  4e-13
    11:  6.5137e-03  6.5137e-03  1e-09  4e-17  4e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1107e-03 -9.5463e-01  6e+01  8e+00  1e+01
     1:  3.4254e-03 -9.1703e-01  4e+00  5e-01  6e-01
     2:  3.4747e-03 -1.7505e-01  3e+00  2e-01  3e-01
     3:  1.4025e-02 -4.0847e-01  1e+00  7e-02  8e-02
     4:  1.5510e-02 -1.2945e-01  2e-01  1e-03  2e-03
     5:  1.5434e-02  2.3711e-03  1e-02  1e-04  1e-04
     6:  9.3490e-03 -2.7641e-03  1e-02  2e-05  3e-05
     7:  8.3867e-03  6.4760e-03  2e-03  4e-06  4e-06
     8:  7.6463e-03  7.0103e-03  6e-04  9e-07  1e-06
     9:  7.2452e-03  7.1776e-03  7e-05  2e-16  8e-13
    10:  7.1915e-03  7.1908e-03  7e-07  1e-16  3e-13
    11:  7.1909e-03  7.1909e-03  7e-09  2e-16  2e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1100e-03 -9.5350e-01  6e+01  8e+00  1e+01
     1:  3.4224e-03 -9.1574e-01  5e+00  5e-01  6e-01
     2:  3.4810e-03 -1.5933e-01  3e+00  2e-01  3e-01
     3:  1.4242e-02 -3.9402e-01  1e+00  7e-02  8e-02
     4:  1.5811e-02 -1.2482e-01  2e-01  1e-03  2e-03
     5:  1.5755e-02  3.6307e-03  1e-02  1e-04  1e-04
     6:  1.0181e-02 -5.2448e-04  1e-02  2e-05  3e-05
     7:  9.0964e-03  7.3014e-03  2e-03  4e-06  4e-06
     8:  8.3461e-03  7.7834e-03  6e-04  9e-07  1e-06
     9:  7.9800e-03  7.9091e-03  7e-05  7e-16  1e-12
    10:  7.9213e-03  7.9206e-03  7e-07  3e-16  2e-13
    11:  7.9207e-03  7.9207e-03  7e-09  7e-17  1e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1092e-03 -9.5236e-01  6e+01  8e+00  1e+01
     1:  3.4194e-03 -9.1444e-01  5e+00  5e-01  6e-01
     2:  3.4878e-03 -1.4350e-01  3e+00  2e-01  3e-01
     3:  1.4462e-02 -3.7901e-01  1e+00  7e-02  9e-02
     4:  1.6114e-02 -1.1837e-01  2e-01  2e-03  2e-03
     5:  1.6076e-02  4.9431e-03  1e-02  1e-04  1e-04
     6:  1.1028e-02  1.6243e-03  9e-03  2e-05  3e-05
     7:  9.8400e-03  8.1952e-03  2e-03  4e-06  4e-06
     8:  9.0127e-03  8.6258e-03  4e-04  5e-07  6e-07
     9:  8.7374e-03  8.6977e-03  4e-05  2e-16  9e-13
    10:  8.7034e-03  8.7030e-03  4e-07  2e-16  1e-13
    11:  8.7031e-03  8.7031e-03  4e-09  2e-16  2e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1085e-03 -9.5123e-01  6e+01  8e+00  1e+01
     1:  3.4164e-03 -9.1315e-01  5e+00  5e-01  6e-01
     2:  3.4951e-03 -1.2757e-01  3e+00  3e-01  3e-01
     3:  1.4683e-02 -3.6349e-01  1e+00  8e-02  9e-02
     4:  1.6419e-02 -1.1012e-01  1e-01  2e-03  2e-03
     5:  1.6398e-02  6.3090e-03  1e-02  1e-04  1e-04
     6:  1.1886e-02  3.6993e-03  8e-03  2e-05  3e-05
     7:  1.0617e-02  9.1545e-03  1e-03  3e-06  4e-06
     8:  9.6858e-03  9.5111e-03  2e-04  1e-16  2e-12
     9:  9.5414e-03  9.5376e-03  4e-06  1e-16  2e-13
    10:  9.5381e-03  9.5380e-03  4e-08  3e-16  2e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1077e-03 -9.5009e-01  6e+01  8e+00  1e+01
     1:  3.4134e-03 -9.1184e-01  5e+00  5e-01  6e-01
     2:  3.5029e-03 -1.1155e-01  3e+00  3e-01  3e-01
     3:  1.4907e-02 -3.4750e-01  1e+00  8e-02  1e-01
     4:  1.6726e-02 -1.0010e-01  1e-01  2e-03  2e-03
     5:  1.6721e-02  7.7300e-03  9e-03  1e-04  1e-04
     6:  1.2751e-02  5.7137e-03  7e-03  2e-05  2e-05
     7:  1.1429e-02  1.0177e-02  1e-03  3e-06  4e-06
     8:  1.0540e-02  1.0407e-02  1e-04  2e-16  2e-12
     9:  1.0430e-02  1.0426e-02  3e-06  1e-16  7e-13
    10:  1.0427e-02  1.0427e-02  8e-08  2e-16  8e-12
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1070e-03 -9.4895e-01  6e+01  8e+00  1e+01
     1:  3.4104e-03 -9.1053e-01  5e+00  5e-01  6e-01
     2:  3.5111e-03 -9.5450e-02  3e+00  3e-01  3e-01
     3:  1.5132e-02 -3.3107e-01  1e+00  8e-02  1e-01
     4:  1.7034e-02 -8.8329e-02  1e-01  2e-03  3e-03
     5:  1.7044e-02  9.2098e-03  8e-03  8e-05  1e-04
     6:  1.3619e-02  7.6776e-03  6e-03  1e-05  2e-05
     7:  1.2280e-02  1.1262e-02  1e-03  2e-06  2e-06
     8:  1.1462e-02  1.1382e-02  8e-05  2e-16  2e-12
     9:  1.1393e-02  1.1391e-02  1e-06  2e-16  7e-13
    10:  1.1392e-02  1.1392e-02  1e-08  1e-16  1e-12
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1062e-03 -9.4780e-01  6e+01  8e+00  1e+01
     1:  3.4074e-03 -9.0922e-01  5e+00  5e-01  6e-01
     2:  3.5198e-03 -7.9276e-02  3e+00  3e-01  3e-01
     3:  1.5360e-02 -3.1424e-01  1e+00  9e-02  1e-01
     4:  1.7344e-02 -7.4866e-02  1e-01  2e-03  3e-03
     5:  1.7367e-02  1.0755e-02  7e-03  3e-05  4e-05
     6:  1.4487e-02  9.5969e-03  5e-03  6e-06  7e-06
     7:  1.3195e-02  1.2360e-02  8e-04  7e-07  9e-07
     8:  1.2468e-02  1.2438e-02  3e-05  1e-16  2e-12
     9:  1.2441e-02  1.2441e-02  3e-07  2e-16  6e-13
    10:  1.2441e-02  1.2441e-02  3e-09  4e-16  8e-13
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1055e-03 -9.4666e-01  6e+01  8e+00  1e+01
     1:  3.4044e-03 -9.0789e-01  5e+00  5e-01  6e-01
     2:  3.5289e-03 -6.3037e-02  3e+00  3e-01  3e-01
     3:  1.5590e-02 -2.9705e-01  2e+00  9e-02  1e-01
     4:  1.7654e-02 -5.9749e-02  1e-01  3e-03  3e-03
     5:  1.7689e-02  1.0476e-02  7e-03  7e-06  9e-06
     6:  1.7228e-02  1.3075e-02  4e-03  3e-06  4e-06
     7:  1.5461e-02  1.1501e-02  4e-03  2e-06  3e-06
     8:  1.4088e-02  1.3473e-02  6e-04  2e-07  3e-07
     9:  1.3732e-02  1.3538e-02  2e-04  1e-16  1e-13
    10:  1.3578e-02  1.3574e-02  4e-06  2e-16  7e-13
    11:  1.3576e-02  1.3576e-02  5e-08  2e-16  7e-12
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1047e-03 -9.4551e-01  6e+01  8e+00  1e+01
     1:  3.4014e-03 -9.0657e-01  5e+00  5e-01  6e-01
     2:  3.5384e-03 -4.6741e-02  3e+00  3e-01  4e-01
     3:  1.5821e-02 -2.7951e-01  2e+00  9e-02  1e-01
     4:  1.7965e-02 -4.2948e-02  1e-01  3e-03  3e-03
     5:  1.8012e-02  9.1798e-03  9e-03  3e-06  4e-06
     6:  1.7846e-02  1.4448e-02  3e-03  1e-06  1e-06
     7:  1.6809e-02  1.3389e-02  3e-03  1e-06  1e-06
     8:  1.5131e-02  1.4715e-02  4e-04  2e-16  5e-14
     9:  1.4855e-02  1.4782e-02  7e-05  1e-16  2e-13
    10:  1.4801e-02  1.4795e-02  6e-06  3e-16  2e-13
    11:  1.4795e-02  1.4795e-02  7e-08  2e-16  1e-12
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1040e-03 -9.4436e-01  6e+01  8e+00  1e+01
     1:  3.3984e-03 -9.0523e-01  5e+00  5e-01  6e-01
     2:  3.5484e-03 -3.0393e-02  3e+00  3e-01  4e-01
     3:  1.6055e-02 -2.6166e-01  2e+00  1e-01  1e-01
     4:  1.8279e-02 -2.4503e-02  8e-02  3e-03  4e-03
     5:  1.8334e-02  8.1025e-03  1e-02  4e-06  4e-06
     6:  1.8268e-02  1.5889e-02  2e-03  8e-07  1e-06
     7:  1.6953e-02  1.4543e-02  2e-03  7e-07  8e-07
     8:  1.6349e-02  1.6083e-02  3e-04  6e-08  8e-08
     9:  1.6105e-02  1.6101e-02  4e-06  6e-10  7e-10
    10:  1.6102e-02  1.6102e-02  4e-08  6e-12  8e-12
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.1032e-03 -9.4321e-01  6e+01  8e+00  1e+01
     1:  3.3954e-03 -9.0389e-01  5e+00  5e-01  7e-01
     2:  3.5588e-03 -1.3999e-02  3e+00  3e-01  4e-01
     3:  1.6290e-02 -2.4351e-01  2e+00  1e-01  1e-01
     4:  1.8596e-02 -4.4529e-03  7e-02  3e-03  4e-03
     5:  1.8655e-02  9.0147e-03  1e-02  2e-05  2e-05
     6:  1.8638e-02  1.7385e-02  1e-03  2e-06  3e-06
     7:  1.7742e-02  1.6587e-02  1e-03  1e-06  1e-06
     8:  1.7581e-02  1.7495e-02  9e-05  9e-08  1e-07
     9:  1.7506e-02  1.7505e-02  2e-06  1e-09  1e-09
    10:  1.7505e-02  1.7505e-02  2e-08  1e-11  1e-11
    Optimal solution found.
         pcost       dcost       gap    pres   dres
     0:  2.3699e-03 -1.4454e+00  7e+01  8e+00  1e+01
     1:  3.8540e-03 -1.3246e+00  1e+00  2e-15  3e-15
     2:  3.7600e-03 -3.6181e-02  4e-02  3e-16  2e-15
     3:  1.2232e-03 -7.7044e-03  9e-03  2e-16  6e-17
     4:  3.0076e-04 -1.9355e-03  2e-03  2e-16  2e-17
     5:  1.5689e-05 -3.7229e-04  4e-04  1e-16  9e-18
     6:  3.5527e-08 -1.4523e-05  1e-05  1e-16  7e-18
     7:  3.6446e-12 -1.4651e-07  1e-07  7e-16  6e-18
     8:  3.6430e-16 -1.4651e-09  1e-09  4e-16  8e-18
    Optimal solution found.


    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:38: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:39: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:43: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:47: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.



```python
plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 18})

plt.plot(aggregate_efficient_frontier.portfolio_stdev,aggregate_efficient_frontier.portfolio_return)


plt.xlabel("portfolio_risk")
plt.ylabel("portfolio_returns")
plt.title("efficient frontier curve")

# plt.yticks(np.linspace(0,45000,11,endpoint=True))
plt.xticks([])
plt.axvline(x=9.163231e-02, linestyle='-.', color='red',label='testing')
plt.text(9.23238e-02, 0.30,'threshold of returns',color='red')
plt.legend(loc='upper left', frameon=False)


plt.show()
```


![png](spend-optimization_files/spend-optimization_33_0.png)



```python
aggregate_efficient_frontier.query("portfolio_return >= 0.269 & portfolio_return <= 0.291")
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
      <th>portfolio_return</th>
      <th>portfolio_stdev</th>
      <th>portfolio_sharpe_ratio</th>
      <th>reject_portfolio_allocation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.26901</td>
      <td>0.000004</td>
      <td>69003.952928</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.27000</td>
      <td>0.001955</td>
      <td>138.086130</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.27100</td>
      <td>0.005880</td>
      <td>46.088226</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.27200</td>
      <td>0.009810</td>
      <td>27.725721</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.27300</td>
      <td>0.013741</td>
      <td>19.867598</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.27400</td>
      <td>0.017676</td>
      <td>15.501273</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.27500</td>
      <td>0.021609</td>
      <td>12.726031</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.27600</td>
      <td>0.025544</td>
      <td>10.805053</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.27700</td>
      <td>0.029478</td>
      <td>9.396979</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.27800</td>
      <td>0.033412</td>
      <td>8.320464</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.27900</td>
      <td>0.037383</td>
      <td>7.463369</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.28000</td>
      <td>0.041550</td>
      <td>6.738945</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.28100</td>
      <td>0.045880</td>
      <td>6.124627</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.28200</td>
      <td>0.050334</td>
      <td>5.602549</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.28300</td>
      <td>0.054878</td>
      <td>5.156894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.28400</td>
      <td>0.059494</td>
      <td>4.773617</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.28500</td>
      <td>0.064165</td>
      <td>4.441687</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.28600</td>
      <td>0.068880</td>
      <td>4.152131</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.28700</td>
      <td>0.073632</td>
      <td>3.897773</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.28800</td>
      <td>0.078413</td>
      <td>3.672879</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.28900</td>
      <td>0.083218</td>
      <td>3.472821</td>
      <td>0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.29000</td>
      <td>0.088046</td>
      <td>3.293747</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### We reject any returns beyond the red vertical --> the risk to reward ratio is not worth it for the amount of additional returns
- looking at returns between 0.27 to 0.290
- lets say we are willing to forego a little returns for lower variance, and we choose to settle at 0.275 (which is our initial return we first run)
- below is the allocation we are interested in


```python
portfolio_weights_sort_weight.query("weightage > 0.00001").head(15)
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
      <th>weightage</th>
      <th>ticker</th>
      <th>nominal_amount_every_1000</th>
      <th>avg_yearly_returns</th>
      <th>yearly_variance</th>
      <th>default_weights</th>
      <th>cumsum_weightage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>0.4399</td>
      <td>LII</td>
      <td>439.949201</td>
      <td>0.306971</td>
      <td>0.037921</td>
      <td>0.017544</td>
      <td>0.4399</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.2370</td>
      <td>CHE</td>
      <td>236.998377</td>
      <td>0.277657</td>
      <td>0.029127</td>
      <td>0.017544</td>
      <td>0.6769</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.1562</td>
      <td>MLAB</td>
      <td>156.173436</td>
      <td>0.281478</td>
      <td>0.064973</td>
      <td>0.017544</td>
      <td>0.8331</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.1071</td>
      <td>MOH</td>
      <td>107.089368</td>
      <td>0.272168</td>
      <td>0.048041</td>
      <td>0.017544</td>
      <td>0.9402</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0.0596</td>
      <td>USPH</td>
      <td>59.600458</td>
      <td>0.268261</td>
      <td>0.016405</td>
      <td>0.017544</td>
      <td>0.9998</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0001</td>
      <td>ICUI</td>
      <td>0.066818</td>
      <td>0.268461</td>
      <td>0.027051</td>
      <td>0.017544</td>
      <td>0.9999</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0001</td>
      <td>KWR</td>
      <td>0.120289</td>
      <td>0.261642</td>
      <td>0.068269</td>
      <td>0.017544</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>



### Cross checking how the top 7 stocks correlates to one another


```python
(
    golden_cluster_correlation_df[['CHE','ICUI','KWR','LII','MLAB','MOH','USPH']]
    .reset_index()
    .query("index in ('CHE','ICUI','KWR','LII','MLAB','MOH','USPH')")
    .rename(columns={"index":"ticker"})
)
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
      <th>ticker</th>
      <th>CHE</th>
      <th>ICUI</th>
      <th>KWR</th>
      <th>LII</th>
      <th>MLAB</th>
      <th>MOH</th>
      <th>USPH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>CHE</td>
      <td>1.000000</td>
      <td>0.741420</td>
      <td>-0.672248</td>
      <td>0.001909</td>
      <td>-0.729724</td>
      <td>0.342591</td>
      <td>-0.665570</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ICUI</td>
      <td>0.741420</td>
      <td>1.000000</td>
      <td>-0.181915</td>
      <td>0.084572</td>
      <td>-0.853767</td>
      <td>-0.273658</td>
      <td>-0.509132</td>
    </tr>
    <tr>
      <th>21</th>
      <td>KWR</td>
      <td>-0.672248</td>
      <td>-0.181915</td>
      <td>1.000000</td>
      <td>0.182174</td>
      <td>0.076818</td>
      <td>-0.395799</td>
      <td>0.212461</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LII</td>
      <td>0.001909</td>
      <td>0.084572</td>
      <td>0.182174</td>
      <td>1.000000</td>
      <td>0.026000</td>
      <td>-0.372286</td>
      <td>-0.130306</td>
    </tr>
    <tr>
      <th>27</th>
      <td>MLAB</td>
      <td>-0.729724</td>
      <td>-0.853767</td>
      <td>0.076818</td>
      <td>0.026000</td>
      <td>1.000000</td>
      <td>-0.058992</td>
      <td>0.654152</td>
    </tr>
    <tr>
      <th>29</th>
      <td>MOH</td>
      <td>0.342591</td>
      <td>-0.273658</td>
      <td>-0.395799</td>
      <td>-0.372286</td>
      <td>-0.058992</td>
      <td>1.000000</td>
      <td>-0.324398</td>
    </tr>
    <tr>
      <th>53</th>
      <td>USPH</td>
      <td>-0.665570</td>
      <td>-0.509132</td>
      <td>0.212461</td>
      <td>-0.130306</td>
      <td>0.654152</td>
      <td>-0.324398</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Thankfully, the top 7 stocks seems to be quite balanced, they are not too heavily tilted towards one correlation

### Closing remarks
- There are some limitations to Markowitz mean-variance portfolio optimization
    1. we assume historical returns and variance to be a good indication moving forward
    1. the returns and variance chosen are static, and this will hugely affect the way the weights of the portfolio chosen

### Next steps?
- This strategy could use some backtesting and see how will it beats market returns across the years from 2012 to 2018
