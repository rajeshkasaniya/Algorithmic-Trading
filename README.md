# Algorithmic-Trading
Implemented 52 week high breakout strategy for Nifty 50 stocks and backtested it on 15 years historical data.

# Introduction: - #
In this project we make an automated trading strategy according to 52-week high breakout trading strategy and then backtested it on the last 15 years’ data for the stocks in Nifty 50 from 2008.

The objective of our project is to implement an Algo trading bot. The advantage of algo trading over manual trading are related to speed, accuracy and reduced costs. Trading with algorithms has the advantage of scanning and executing on multiple indicators at a speed that no human could do. One of the biggest advantages of the algo trading is the ability to remove human emotion from the markets, as trades are constrained within a set of predefined criteria.

To make this project we use python and their libraries like pandas, numpy etc. And to get the data we used Yahoo Finance.

For this strategy I made some rule for trading –
-	Buy when the stock recent weekly closing price is above the last 52 week closing price.
-	Then our initial Stop loss will be 30%.
-	And our target will be 300%. So basically for trade we take initial Risk: reward ratio of 1:10.
-	To protect the gains when stock moves in our direction, I used Trailing Stop Loss of last 25-week low.
-	My initial Capital was 80lac. And in any trade I used only 1.25% of capital. So there will be very less risk taken in every trade. That will reduce our profit but it will insure that there will not be too much effect on our account size when we will make a wrong trade.
 
# Results: - #

Following results, we get:
-	We make profit of 390lac during the last 13 years’ period.
-	So our absolute return was 481.2 % during that period.
-	Average annual return was 37.1 %.
-	CAGR for that period was 14.5%.
-	Sharpe ratio was 0.82.
-	Max. loss in one year -20.06% in 2020.
-	Max. profit in one year 145% in 2018.
-	Top performing Stock is BajFinance which give profit of 133Lac.
-	Worst performing stock was Unitech which give 87k loss.


# Conclusion:- #

-	In this project we make an algo trading bot , which will automate our trading process. 
-	By using this we can eliminate our emotion from trading and we can execute our trades faster then we can think. We can also visualize our profit and loss more clearly on every trade.
-	We can save a lot of time by using this because now we need not to stay in front of laptop screen to see charts and make decision because now our bot make decision according to our strategy.
-	Thus, this is a very useful project in any trader’s life.
