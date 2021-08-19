# stockmarketdatacollection

## *Data Pipelines for Gathering Stock Market Data*

Interested in gathering stock market data? 

This repository aims to be a resource for those that want to capture these data and store in a PostgreSQL database.

There are three files in this repository apart from the repository basics - two Python files (GrabStockData, dataconfig) and a SQL file (initializestockdata).

The SQL file creates the *stockdata* database and *tdameritrade_stocks* table necessary to store data that can be captured through the Python files. This database and table should be initialized prior to working with the Python scripts. If you are unsure of how to get started with PostgreSQL, I recommend visiting the main [website](https://www.postgresql.org/).

The Python files work together to grab these data from the TD Ameritrade API after the markets have closed for the day. Please be sure you fill in the appropriate information in the dataconfig file specific to your computer/TD Ameritrade credentials so items run without error. If you are unsure on how to generate your TD Ameritrade credentials, I highly encourage you to watch this [video](https://www.youtube.com/watch?v=P5YanfJFlNs&list=RDCMUCY2ifv8iH1Dsgjrz-h3lWLQ&start_radio=1&rv=P5YanfJFlNs&t=0). He gives excellent overviews of how to get set up and the incredible *tda-api* Python package.

After everything is configured in the dataconfig file, you need to modify (in the GrabStockData file) is the file path to where the dataconfig file is so it can be properly imported. After that, make sure all the necessary packages are installed for the Python scripts and schedule a job to kick of the GrabStockData file after the markets close and start collecting data! 

As a quick side note - your ticker list (path specified in dataconfig file) should be imported as a dataframe with the column name "symbol" to be properly incorporated into the pipeline.

A big thanks to the Kaggle user KratiSaxena for the excellent [notebook](https://www.kaggle.com/kratisaxena/stock-market-technical-indicators-visualization) about technical indicator generation in Python! This was a fantastic resource, and I did not make many edits to the functions to get them to my liking.

Lastly, note that this script was written/tested on a Macbook, so if Linux or Windows users are having issues, please feel free to contact me and we can troubleshoot.
