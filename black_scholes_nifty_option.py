from quantmod.derivatives import OptionData
from datetime import datetime, timedelta
from quantmod.models import OptionInputs, BlackScholesOptionPricing

# Get option chain for specify expiry
expiration = '27-Mar-2025'
valuation = datetime.today()
ttm = (pd.to_datetime(expiration+' 15:30:00') - pd.to_datetime(valuation)) / timedelta(days=365)

# Instantiate the Option Data
opt = OptionData("NIFTY", expiration)
df = opt.get_call_option_data
df.head(2)

# querry strikes between 22000 and 22500
df = df.query('strikePrice>=22000 and strikePrice<=23000').reset_index(drop=True)
df.head()

# Dataframe manipulation with selected fields
df1 = pd.DataFrame({'Strike': df['strikePrice'],
                    'Price': df['lastPrice'],
                    })

# Instantiate BS Pricing Engine from quantmod and Derive Implied Volatiliy
for i in range(len(df1)):
    nifty = BlackScholesOptionPricing(
        OptionInputs(
            spot = 22400,
            strike = df1['Strike'].iloc[i], 
            rate = 0.0,
            ttm = ttm,
            volatility = 0.20,
            callprice = df1['Price'].iloc[i]
            )
        )
        
    df1.loc[i, 'ImpVol'] = nifty.impvol
    
# Check output
df1.head(10)

# Derive greeks and assign to dataframe as columns
for i in range(len(df1)):
    # initializing the BS option object
    nifty = BS(
        spot = 22400, 
        strike = df1['Strike'].iloc[i], 
        rate = 0.0,
        dte = ttm, 
        volatility=df1['ImpVol'].iloc[i]
        )
    # assign greeks to dataframe 
    df1.loc[i, 'Delta'] = nifty.call_delta
    df1.loc[i, 'Gamma'] = nifty.gamma
    df1.loc[i, 'Vega'] = nifty.vega
    df1.loc[i, 'Theta'] = nifty.call_theta
    
# Verify output
df1.head(10)

# Plot graph iteratively
fig, axes = plt.subplots(2, 2, figsize=(20, 10))

# Greek parameters
greeks = {
    (0,0): ('Delta', 'r'),
    (0,1): ('Gamma', 'b'),
    (1,0): ('Vega', 'k'),
    (1,1): ('Theta', 'g')
}

# Plot all greeks in one loop
for (i,j), (greek, color) in greeks.items():
    axes[i,j].plot(df1['Strike'], df1[greek], color=color, label=expiration)
    axes[i,j].set_title(greek)
    axes[i,j].legend()

plt.show()