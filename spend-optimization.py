
# coding: utf-8

# ___
# due to sensitive information, actual real numbers will not be used here. instead it will be substituted with real stocks
# 
# This notebook will show the concept of spend optimization based on returns and variance. It will answer the question of:
# 1. if you have $100, where will you spend it on?
# 1. the spend allocation should give us the maximum returns and lowest variance



get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime 
from dateutil.relativedelta import relativedelta
import cvxopt as opt
from cvxopt import blas, solvers, matrix


# ### Context of intention of analysis
# - Lets say we have `$100`, where should we spend on paid acquisition of users?
# - So that ROI will be more than current and also take on lesser risk
# - This becomes a 'portfolio' optimization problem, and each channel and geo can be considered as 1 stock. We will end up with a lot of options to choose from
#     - Lets use `YouTube` for example. `YouTube` singapore will be considered as 1 stock
# - And we will choose a bunch of stocks that will return us the maximum returns and lowest variance
# - Here, we will not be using the real data i had, but instead use real stock information to illustrate the idea

# ### Scope of analysis
# - Data pull
# - Generate covariance and correlation matrix
# - Pick stocks based on optimization
# - Get portfolio return and variance (and sharpe ratio)
# - Plot efficient frontier

# ### Data pull
# - the data used will be from the previous blog post of k-means: [link](https://medium.com/uptick-blog/stock-picks-using-k-means-clustering-4330c6c4e8de)
# - we will be focusing on the golden cluster identified, which has 57 stocks with an average of ~24% return and ~5% variance



rawdata = (
    pd.read_csv('data/aggregate_tickers_return_variance.csv', index_col=0)
    .reset_index()
)

rawdata_yearly_returns = (
    pd.read_csv("data/yearly_returns_for_tickers.csv")
)


# #### Query the golden cluster stocks only



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


# ### Maximum returns and variance + mininum returns and variance of stocks
# - this sets the maximum and minimum boundary of both returns and variance of portfolio



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


# ### Covariance matrix
# - diagonals are variance of stocks
# - Covariance is a measure of the directional relationship between the returns on two assets.
#     - Positive covariance means that both returns move together while a negative covariance means returns move inversely



golden_cluster_covariance_df = (
    rawdata_yearly_returns
    .drop(['year'],1)
    .cov()
)

golden_cluster_covariance_df.head()


# ### Correlation matrix
# - $\rho = \frac{\sigma_{xy}}{\sigma_x \sigma_y} = \frac{covariance}{\sigma_x \sigma_y}
# $
# - this tells us how each stock return correlate with each other
# - ideally, we would like our portfolio to have a good balance of negative and positive correlation



sigma_x_sigma_y = (
#     pd.DataFrame(
        np.dot(
            (np.array([np.sqrt(golden_cluster_stocks.yearly_variance)]).T),
            np.array([np.sqrt(golden_cluster_stocks.yearly_variance)])
        )
#     )
)




golden_cluster_correlation_df = golden_cluster_covariance_df/sigma_x_sigma_y
golden_cluster_correlation_df.head()


# ### Pick stocks based on optimization
# 
# #### Idea of portfolio optimization
# - Ideally, we would want our portfolio to have high returns and low risk (standard deviation)
# - Assuming we have a list of stocks to pick, with historical information of returns and risk, is it possible to pick stocks with different weightage to have returns higher than average and risk lower than average?
# - Yes, markowitz mean-variance portfolio optimization allows us to achieve just the above
# - Intuitively without any mathematical formula introduced, we would like to: 
#     1. minimize variance 
#     1. achieve a set amount of returns
#     1. sum of weights of all stock = 1
#     1. we assume no short position, so weightage of one stock >= 0 and <= 1
#     
#     
# - Now if we have to place the objective function, it will be the portfolio variance:
#     - $ w^T * Cov_{matrix} * w$
# - subjected to the following constraints:
#     - portfolio returns = $ \sum (w_i *\mu_i) $ = R
#     - sum of weights = $\sum w_i$ = 1
#     - weight of each stock = $ 0 \leq w_i \leq 1$
# 
# 
# - if we think a little more about the covariance formula, $ \frac{1}{n} \sum((x_i - \mu_x)(y_i - \mu_y))$
#     - it is a non-linear equation but more specifically a quadratic equation
#     - which means there is a specific solution if constraints are fulfilled
#     - we can go about solving this with the convex approach



golden_cluster_covariance_matrix = matrix(golden_cluster_covariance_df.as_matrix(),(57,57))




golden_cluster_return_matrix = (
    matrix(
        golden_cluster_stocks
        [['avg_yearly_returns']]
        .as_matrix(), (57,1)
    )
)


# ### Initialize optimization function for weights allocation



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


# ### Get weights of portfolio
# - `r_min` is set at 15% higher than the default average of golden cluster
# - hopefully the variance will be lower than the average of the golden cluster



solution = optimize_portfolio(57, golden_cluster_return_matrix, golden_cluster_covariance_matrix, 0.275)


# ### Dataframe of portfolio weights



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


# ### Portfolio returns, standard deviation and sharpe ratio
# - sharpe ratio is a measure of risk-to-reward ratio, $sharpeRatio = \frac{portfolioReturns}{portfolioStdDev}$
#     - a higher magnitude the better it is



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




print("portfolio returns: " + str(round(float(portfolio_returns),4)))
print("portfolio standard deviation: " + str(round(float((portfolio_stdev)),4))) 
print("portfolio sharpe ratio: " + str(round(float((portfolio_sharpe_ratio)),4))) 




portfolio_weights_sort_weight = (
    portfolio_weights
    .sort_values(['weightage'],ascending=False)
    .pipe(lambda x:x.assign(cumsum_weightage = x.weightage.cumsum()))
)


# ### Construct efficient frontier
# - assuming we optimize the returns for each possible scenario



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
    .query("portfolio_return < 0.24 | portfolio_return >= 0.26")
)




plt.figure(figsize=(20,10))

plt.plot(aggregate_efficient_frontier.portfolio_stdev,aggregate_efficient_frontier.portfolio_return)

plt.legend(loc='upper left', frameon=False)
plt.xlabel("portfolio_risk")
plt.ylabel("portfolio_returns")
plt.title("efficient frontier curve")

# plt.yticks(np.linspace(0,45000,11,endpoint=True))
plt.xticks([])
plt.axvline(x=9.163231e-02, linestyle='-.', color='red',label='testing')
plt.text(9.23238e-02, 0.30,'threshold of returns',color='red')

plt.show()


# ### We reject any returns beyond the red vertical --> the risk to reward ratio is not worth it for the amount of additional returns
# - looking at returns between 0.27 to 0.290
# - lets say we are willing to forego a little returns for lower variance, and we choose to settle at 0.275 (which is our initial return we first run)
# - below is the allocation we are interested in



portfolio_weights_sort_weight.query("weightage > 0.00001").head(15)


# ### Cross checking how the top 7 stocks correlates to one another



(
    golden_cluster_correlation_df[['CHE','USPH','MLAB','KWR','UNH','LII','MOH']]
    .reset_index()
    .query("index in ('CHE','USPH','MLAB','KWR','UNH','LII','MOH')")
    .rename(columns={"index":"ticker"})
)


# #### Thankfully, the top 7 stocks seems to be quite balanced, they are not too heavily tilted towards one correlation

# ### Closing remarks
# - There are some limitations to Markowitz mean-variance portfolio optimization
#     1. we assume historical returns and variance to be a good indication moving forward
#     1. the returns and variance chosen are static, and this will hugely affect the way the weights of the portfolio chosen

# ### Next steps?
# - This strategy could use some backtesting and see how will it beats market returns across the years from 2012 to 2018
