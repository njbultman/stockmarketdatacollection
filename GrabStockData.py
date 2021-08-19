"""
Script to run daily after market hours and capture data from TD Ameritrade API & store in PostgreSQL database
@author: Nick Bultman, August 2021
"""
import pandas as pd
import os
import numpy as np
import datetime
import pandas_market_calendars as mcal
import pytz
import time
import random
from tda import auth, client
import copy
import psycopg2

# Before going any further, check to make sure today is a day the market is open
if mcal.get_calendar('NYSE').valid_days(start_date = str((pytz.timezone('America/Chicago').localize((datetime.datetime.now()))).date()), end_date = str((pytz.timezone('America/Chicago').localize((datetime.datetime.now()))).date())).empty == True:
    print(str(datetime.datetime.now()) + ': Market closed today. Halting code execution.')
    
else:

    # Change directory and import dataconfig file
    os.chdir('/path/to/dataconfig/here')

    import dataconfig

    # Grab NASDAQ & non-NASDAQ symbols
    Symbols_py = pd.read_csv(dataconfig.symbolpath).symbol.to_list()
    
    # Select start and end time for pulling historical data
    end = datetime.datetime.now()
    start = mcal.get_calendar('NYSE').valid_days(end_date = str((pytz.timezone('America/Chicago').localize((datetime.datetime.now()))).date()), start_date = str((pytz.timezone('America/Chicago').localize((datetime.datetime.now()))).date() - datetime.timedelta(days = 60)))[-28].to_pydatetime().replace(tzinfo = None)

    # Get TD Ameritrade Credentials
    token_path = dataconfig.token_path
    api_key = dataconfig.api_key
    redirect_uri = dataconfig.redirect_uri

    try:
        c = auth.client_from_token_file(token_path, api_key)
    except FileNotFoundError:
        from selenium import webdriver
        with webdriver.Chrome(executable_path = dataconfig.webdriverpath) as driver:
            c = auth.client_from_login_flow(
                driver, api_key, redirect_uri, token_path)

    # Split ticker list into multiple lists of 100 since TD Ameritrade API is throttled
    ticker_batches = [Symbols_py[x:x+100] for x in range(0, len(Symbols_py), 100)]

    print('Pulling latest quotes at: ' + str(datetime.datetime.now()))
    
    total_df_final = pd.DataFrame()

    # Grab quotes
    for i in ticker_batches:
        
        quotes = c.get_quotes(i).json()
        
        for i in quotes.keys():
            
            try:
                quote = pd.DataFrame([quotes[i]])
                quote_df = pd.DataFrame(quote.values, columns = quote.columns)
            except:
                quote_df = pd.DataFrame(columns = quote.columns)
                quote_df = pd.concat([quote_df, pd.Series(np.nan)])
                quote_df.symbol = i
                
            total_df_final = pd.concat([total_df_final, quote_df], axis = 0)
    
    # Add today's date and clean before pulling historical data
    total_df_final['date'] = str(datetime.date.today())

    total_df_final.reset_index(inplace = True)

    total_df_final.drop('index', inplace = True, axis = 1)
    
    print(str(datetime.datetime.now()) + ': Pulling historical data.')
    
    # Grab historical stock price data to generate technical indicators
    history_data = pd.DataFrame()
    
    for i in total_df_final.symbol.unique():
        try:
            ticker_raw = pd.DataFrame(c.get_price_history(i,
                                                            period_type=client.Client.PriceHistory.PeriodType.YEAR,
                                                            period=client.Client.PriceHistory.Period.ONE_YEAR,
                                                            frequency_type=client.Client.PriceHistory.FrequencyType.DAILY,
                                                            frequency=client.Client.PriceHistory.Frequency.DAILY).json()['candles'])
            ticker_raw.datetime = [datetime.datetime.fromtimestamp(x / 1000) for x in ticker_raw.datetime]
            ticker_raw = ticker_raw.loc[(ticker_raw.datetime >= start) & (ticker_raw.datetime < end)]
            ticker_raw['Label'] = i
            ticker_raw.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Label']
            ticker_raw = ticker_raw[['Date', 'Label', 'Close', 'High', 'Low', 'Open', 'Volume']]     
            history_data = pd.concat([history_data, ticker_raw], axis = 0)
            time.sleep(1)
        except:
            None
        
    # Append current day quote information to be used for generating technical indicators
    history_data_now = total_df_final[['date', 'symbol', 'regularMarketLastPrice', 'highPrice', 'lowPrice', 'openPrice', 'totalVolume']]
    history_data_now = history_data_now.rename({'date': 'Date', 'symbol': 'Label', 'regularMarketLastPrice': 'Close', 'highPrice': 'High', 'lowPrice': 'Low', 'openPrice': 'Open', 'totalVolume': 'Volume'}, axis = 1)
    history_data = pd.concat([history_data, history_data_now], axis = 0)
    history_data.Date = pd.to_datetime(history_data.Date)
    
    # Add any missing weekday dates that are not banking holidays and populate values with 0
    history_data.index = pd.DatetimeIndex(history_data.Date)
    history_data.drop('Date', axis = 1, inplace = True)
    history_data = history_data.groupby('Label').resample('D').sum()
    history_data.drop('Label', axis = 1, inplace = True)
    history_data = history_data.reset_index(level = 0)
    valid_days = mcal.get_calendar('NYSE').valid_days(end_date = str((pytz.timezone('America/Chicago').localize((datetime.datetime.now()))).date()), start_date = str((pytz.timezone('America/Chicago').localize((datetime.datetime.now()))).date() - datetime.timedelta(days = 60)))
    history_data = history_data[(history_data.index.dayofweek < 5) & history_data.index.isin(valid_days)]
    history_data.reset_index(inplace = True)
    
    print(str(datetime.datetime.now()) + ': Generating technical indicators.')
    
    # Create a list of dataframes for technical indicator creation
    TechIndicator = [
    
        TechIndicator 
    
        for _, TechIndicator in history_data.groupby('Label')
    
    ]
    
    [df.reset_index(inplace=True) for df in TechIndicator]
    [df.drop('index', axis = 1) for df in TechIndicator]
    
    # Generate technical indicators
    # Momentum 1D & RSI
    # Momentum_1D = P(t) - P(t-1)
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['Momentum_1D'] = (TechIndicator[stock]['Close']-TechIndicator[stock]['Close'].shift(1))
        TechIndicator[stock]['RSI_14D'] = TechIndicator[stock]['Momentum_1D'].rolling(center=False, window=14).apply(dataconfig.rsi)
    
    # Aroon Oscillator
    for stock in range(len(TechIndicator)):
        listofzeros = [0] * 25
        up, down = dataconfig.aroon(TechIndicator[stock])
        aroon_list = [x - y for x, y in zip(up,down)]
        if len(aroon_list)==0:
            aroon_list = [0] * TechIndicator[stock].shape[0]
            TechIndicator[stock]['Aroon_Oscillator'] = aroon_list
        else:
            TechIndicator[stock]['Aroon_Oscillator'] = listofzeros+aroon_list
    
    # Acceleration Bands
    for stock in range(len(TechIndicator)):
        dataconfig.abands(TechIndicator[stock])
        TechIndicator[stock] = TechIndicator[stock]
    
    # Stochastic Oscillator (%K & %D)
    for stock in range(len(TechIndicator)):
        dataconfig.STOK(TechIndicator[stock], 4)
    
    # Parabolic SAR
    for stock in range(len(TechIndicator)):
        dataconfig.psar(TechIndicator[stock])
    
    # Price ROC
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['ROC'] = ((TechIndicator[stock]['Close'] - TechIndicator[stock]['Close'].shift(12))/(TechIndicator[stock]['Close'].shift(12)))*100
    
    # Momentum
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['Momentum'] = TechIndicator[stock]['Close'] - TechIndicator[stock]['Close'].shift(4)
    
    # EWA and Triple EMA
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['EMA'] = TechIndicator[stock]['Close'].ewm(span=3,min_periods=0,adjust=True,ignore_na=False).mean()
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['TEMA'] = (3 * TechIndicator[stock]['EMA'] - 3 * TechIndicator[stock]['EMA'] * TechIndicator[stock]['EMA']) + (TechIndicator[stock]['EMA']*TechIndicator[stock]['EMA']*TechIndicator[stock]['EMA'])
    
    # Normalized Average True Range
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['HL'] = TechIndicator[stock]['High'] - TechIndicator[stock]['Low']
        TechIndicator[stock]['absHC'] = abs(TechIndicator[stock]['High'] - TechIndicator[stock]['Close'].shift(1))
        TechIndicator[stock]['absLC'] = abs(TechIndicator[stock]['Low'] - TechIndicator[stock]['Close'].shift(1))
        TechIndicator[stock]['TR'] = TechIndicator[stock][['HL','absHC','absLC']].max(axis=1)
        TechIndicator[stock]['ATR'] = TechIndicator[stock]['TR'].rolling(window=14).mean()
        TechIndicator[stock]['NATR'] = (TechIndicator[stock]['ATR'] / TechIndicator[stock]['Close']) *100
    
    # Average Directional Moving Index
    for stock in range(len(TechIndicator)):
        dataconfig.DMI(TechIndicator[stock], 14)
    
    # MACD
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['26_ema'] = TechIndicator[stock]['Close'].ewm(span=26,min_periods=0,adjust=True,ignore_na=False).mean()
        TechIndicator[stock]['12_ema'] = TechIndicator[stock]['Close'].ewm(span=12,min_periods=0,adjust=True,ignore_na=False).mean()
        TechIndicator[stock]['MACD'] = TechIndicator[stock]['12_ema'] - TechIndicator[stock]['26_ema']
    
    # Money Flow Index
    for stock in range(len(TechIndicator)):
        dataconfig.MFI(TechIndicator[stock])
        
    # Ichimoku Cloud
    for stock in range(len(TechIndicator)):
        dataconfig.ichimoku(TechIndicator[stock])
        
    # William %R
    for stock in range(len(TechIndicator)):
        dataconfig.WillR(TechIndicator[stock])
        
    # Adaptive Moving Average
    for stock in range(len(TechIndicator)):
        TechIndicator[stock]['KAMA'] = dataconfig.KAMA(TechIndicator[stock]['Close'])
    
    print(str(datetime.datetime.now()) + ': Combining data and cleaning.')
    
    # Combine all list dataframes into one dataframe
    history_data_man = pd.DataFrame()
    
    for stock in range(len(TechIndicator)):
        history_data_man = pd.concat([history_data_man, TechIndicator[stock]], axis = 0)
    
    history_data_man = history_data_man.replace([np.inf, -np.inf], np.nan)
    
    # Calculate return before dropping NA and filtering
    history_data_man['daily_return'] = history_data_man.groupby('Label').Close.pct_change()
    
    filter_date = pd.to_datetime(end)
    
    # Filter for end date
    history_data_man = history_data_man.loc[history_data_man.Date == datetime.date(filter_date.year, filter_date.month, filter_date.day)]
    
    # Reset index
    history_data_man.reset_index(inplace = True)
    
    # Drop unnecessary columns
    history_data_man.drop(['level_0', 'index', 'Date'], axis = 1, inplace = True)
    
    # Merge with TDA bid data
    total_df_final = total_df_final.merge(history_data_man, how = 'left', left_on = 'symbol',
                         right_on = 'Label')
    
    total_df_final.drop(['Label', 'Close', 'High', 'Low', 'Open', 'Volume'], axis = 1, inplace = True)
    
    total_df_final = total_df_final.replace(np.inf, 0)
    
    # Convert dates to datetime from strings
    total_df_final.date = pd.to_datetime(total_df_final.date)
    total_df_final.divDate = pd.to_datetime(total_df_final.divDate)
    
    # Convert blanks in bidTick to NULL
    total_df_final.bidTick = total_df_final.bidTick.str.replace(' ', 'NULL')
    
    # Change columns for SQL database
    total_df_final.rename(columns={'52WkHigh':'WkHigh52', '52WkLow':'WkLow52', '26_ema':'ema_26', '12_ema':'ema_12'}, inplace=True)
    
    # Convert apostrophe's in description column to blank to not error SQL query
    total_df_final.description = total_df_final.description.str.replace("'", "")
    
    print(str(datetime.datetime.now()) + ': Beginning insertion into stockdata database (tdameritrade_stocks table).')
    
    # Insert contents into stock_data database (tdameritrade_stocks table)
    conn = psycopg2.connect(
            database = dataconfig.db,
            user = dataconfig.dbuser,
            password = dataconfig.dbpassword,
            host = dataconfig.dbhost,
            port = dataconfig.dbport
            )
    
    cur = conn.cursor()
    for i in range(len(total_df_final)):
        sqlstring = "INSERT INTO tdameritrade_stocks( \
                            symbol, \
                            date, \
                            WkHigh52, \
                            WkLow52, \
                            askId, \
                            askPrice, \
                            askSize, \
                            assetMainType, \
                            assetSubType, \
                            assetType, \
                            bidId, \
                            bidPrice, \
                            bidSize, \
                            bidTick, \
                            closePrice, \
                            cusip, \
                            delayed, \
                            description, \
                            digits, \
                            divAmount, \
                            divDate, \
                            divYield, \
                            exchange, \
                            exchangeName, \
                            highPrice, \
                            lastId, \
                            lastPrice, \
                            lastSize, \
                            lowPrice, \
                            marginable, \
                            mark, \
                            markChangeInDouble, \
                            markPercentChangeInDouble, \
                            nAV, \
                            netChange, \
                            netPercentChangeInDouble, \
                            openPrice, \
                            peRatio, \
                            quoteTimeInLong, \
                            realtimeEntitled, \
                            regularMarketLastPrice, \
                            regularMarketLastSize, \
                            regularMarketNetChange, \
                            regularMarketPercentChangeInDouble, \
                            regularMarketTradeTimeInLong, \
                            securityStatus, \
                            shortable, \
                            totalVolume, \
                            tradeTimeInLong, \
                            volatility, \
                            Momentum_1D, \
                            RSI_14D, \
                            Aroon_Oscillator, \
                            AB_Middle_Band, \
                            aupband, \
                            AB_Upper_Band, \
                            adownband, \
                            AB_Lower_Band, \
                            STOK, \
                            STOD, \
                            psar, \
                            ROC, \
                            Momentum, \
                            EMA, \
                            TEMA, \
                            HL, \
                            absHC, \
                            absLC, \
                            TR, \
                            ATR, \
                            NATR, \
                            UpMove, \
                            DownMove, \
                            Zero, \
                            PlusDM, \
                            MinusDM, \
                            plusDI, \
                            minusDI, \
                            ADX, \
                            ema_26, \
                            ema_12, \
                            MACD, \
                            tp, \
                            rmf, \
                            pmf, \
                            nmf, \
                            mfr, \
                            Money_Flow_Index, \
                            turning_line, \
                            standard_line, \
                            ichimoku_span1, \
                            ichimoku_span2, \
                            chikou_span, \
                            WillR, \
                            KAMA, \
                            daily_return) VALUES (" + \
                            "'" + str(total_df_final.symbol[i]) +  "'" + "," + \
                            "'" + str(datetime.date(total_df_final.date[i].year, total_df_final.date[i].month, total_df_final.date[i].day)) + "'" + "," + \
                            str(total_df_final['WkHigh52'][i]) + "," + \
                            str(total_df_final['WkLow52'][i]) + "," + \
                            "'" + str(total_df_final.askId[i]) +  "'" + "," + \
                            str(total_df_final['askPrice'][i]) + "," + \
                            str(total_df_final['askSize'][i]) + "," + \
                            "'" + str(total_df_final.assetMainType[i]) +  "'" + "," + \
                            "'" + str(total_df_final.assetSubType[i]) +  "'" + "," + \
                            "'" + str(total_df_final.assetType[i]) +  "'" + "," + \
                            "'" + str(total_df_final.bidId[i]) +  "'" + "," + \
                            str(total_df_final['bidPrice'][i]) + "," + \
                            str(total_df_final['bidSize'][i]) + "," + \
                            str(total_df_final['bidTick'][i]) + "," + \
                            str(total_df_final['closePrice'][i]) + "," + \
                            "'" + str(total_df_final.cusip[i]) +  "'" + "," + \
                            str(total_df_final['delayed'][i]) + "," + \
                            "'" + str(total_df_final.description[i]) +  "'" + "," + \
                            str(total_df_final['digits'][i]) + "," + \
                            str(total_df_final['divAmount'][i]) + "," + \
                            "'" + str(total_df_final['divDate'][i]) + "'" + "," + \
                            str(total_df_final['divYield'][i]) + "," + \
                            "'" + str(total_df_final.exchange[i]) +  "'" + "," + \
                            "'" + str(total_df_final.exchangeName[i]) +  "'" + "," + \
                            str(total_df_final['highPrice'][i]) + "," + \
                            "'" + str(total_df_final.lastId[i]) +  "'" + "," + \
                            str(total_df_final['lastPrice'][i]) + "," + \
                            str(total_df_final['lastSize'][i]) + "," + \
                            str(total_df_final['lowPrice'][i]) + "," + \
                            str(total_df_final['marginable'][i]) + "," + \
                            str(total_df_final['mark'][i]) + "," + \
                            str(total_df_final['markChangeInDouble'][i]) + "," + \
                            str(total_df_final['markPercentChangeInDouble'][i]) + "," + \
                            str(total_df_final['nAV'][i]) + "," + \
                            str(total_df_final['netChange'][i]) + "," + \
                            str(total_df_final['netPercentChangeInDouble'][i]) + "," + \
                            str(total_df_final['openPrice'][i]) + "," + \
                            str(total_df_final['peRatio'][i]) + "," + \
                            str(total_df_final['quoteTimeInLong'][i]) + "," + \
                            str(total_df_final['realtimeEntitled'][i]) + "," + \
                            str(total_df_final['regularMarketLastPrice'][i]) + "," + \
                            str(total_df_final['regularMarketLastSize'][i]) + "," + \
                            str(total_df_final['regularMarketNetChange'][i]) + "," + \
                            str(total_df_final['regularMarketPercentChangeInDouble'][i]) + "," + \
                            str(total_df_final['regularMarketTradeTimeInLong'][i]) + "," + \
                            "'" + str(total_df_final.securityStatus[i]) +  "'" + "," + \
                            str(total_df_final['shortable'][i]) + "," + \
                            str(total_df_final['totalVolume'][i]) + "," + \
                            str(total_df_final['tradeTimeInLong'][i]) + "," + \
                            str(total_df_final['volatility'][i]) + "," + \
                            str(total_df_final['Momentum_1D'][i]) + "," + \
                            str(total_df_final['RSI_14D'][i]) + "," + \
                            str(total_df_final['Aroon_Oscillator'][i]) + "," + \
                            str(total_df_final['AB_Middle_Band'][i]) + "," + \
                            str(total_df_final['aupband'][i]) + "," + \
                            str(total_df_final['AB_Upper_Band'][i]) + "," + \
                            str(total_df_final['adownband'][i]) + "," + \
                            str(total_df_final['AB_Lower_Band'][i]) + "," + \
                            str(total_df_final['STOK'][i]) + "," + \
                            str(total_df_final['STOD'][i]) + "," + \
                            str(total_df_final['psar'][i]) + "," + \
                            str(total_df_final['ROC'][i]) + "," + \
                            str(total_df_final['Momentum'][i]) + "," + \
                            str(total_df_final['EMA'][i]) + "," + \
                            str(total_df_final['TEMA'][i]) + "," + \
                            str(total_df_final['HL'][i]) + "," + \
                            str(total_df_final['absHC'][i]) + "," + \
                            str(total_df_final['absLC'][i]) + "," + \
                            str(total_df_final['TR'][i]) + "," + \
                            str(total_df_final['ATR'][i]) + "," + \
                            str(total_df_final['NATR'][i]) + "," + \
                            str(total_df_final['UpMove'][i]) + "," + \
                            str(total_df_final['DownMove'][i]) + "," + \
                            str(total_df_final['Zero'][i]) + "," + \
                            str(total_df_final['PlusDM'][i]) + "," + \
                            str(total_df_final['MinusDM'][i]) + "," + \
                            str(total_df_final['plusDI'][i]) + "," + \
                            str(total_df_final['minusDI'][i]) + "," + \
                            str(total_df_final['ADX'][i]) + "," + \
                            str(total_df_final['ema_26'][i]) + "," + \
                            str(total_df_final['ema_12'][i]) + "," + \
                            str(total_df_final['MACD'][i]) + "," + \
                            str(total_df_final['tp'][i]) + "," + \
                            str(total_df_final['rmf'][i]) + "," + \
                            str(total_df_final['pmf'][i]) + "," + \
                            str(total_df_final['nmf'][i]) + "," + \
                            str(total_df_final['mfr'][i]) + "," + \
                            str(total_df_final['Money_Flow_Index'][i]) + "," + \
                            str(total_df_final['turning_line'][i]) + "," + \
                            str(total_df_final['standard_line'][i]) + "," + \
                            str(total_df_final['ichimoku_span1'][i]) + "," + \
                            str(total_df_final['ichimoku_span2'][i]) + "," + \
                            str(total_df_final['chikou_span'][i]) + "," + \
                            str(total_df_final['WillR'][i]) + "," + \
                            str(total_df_final['KAMA'][i]) + "," + \
                            str(total_df_final['daily_return'][i]) + ")"
        sqlstring = sqlstring.replace('nan', 'NULL')
        sqlstring = sqlstring.replace('NaT', 'NULL')
        sqlstring = sqlstring.replace("'NULL'", 'NULL')
        cur.execute(sqlstring)
        conn.commit()
    
    print(str(datetime.datetime.now()) + ': Process successfully completed.')
    
    
    
    