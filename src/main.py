import metric
from financeIO import GoogleFinanceIO
import datetime
import pandas as pd
import label

#done 1: download 1-day data
#todo: semi-done 2: calculate metrics if available
#todo 3: maybe cluster features
#done 4: calculate labels
#todo 5: create images if available
#todo 6: train & validation
#todo 7: test
#todo 8: plot

DATE_FORMAT = '%d-%m-%Y'

finance_io = GoogleFinanceIO()

metric_handler = metric.Metric()

metric_handler.add(metric.RSI(15))
metric_handler.add(metric.RSI(20))
metric_handler.add(metric.RSI(25))
metric_handler.add(metric.RSI(30))

metric_handler.add(metric.SMA(15))
metric_handler.add(metric.SMA(20))
metric_handler.add(metric.SMA(25))
metric_handler.add(metric.SMA(30))

metric_handler.add(metric.MACD(26,12))
metric_handler.add(metric.MACD(28,14))
metric_handler.add(metric.MACD(30,16))

metric_handler.add(metric.MACD_Trigger(9,26,12))
metric_handler.add(metric.MACD_Trigger(10,28,14))
metric_handler.add(metric.MACD_Trigger(11,30,16))

metric_handler.add(metric.WilliamR(14))
metric_handler.add(metric.WilliamR(18))
metric_handler.add(metric.WilliamR(22))

metric_handler.add(metric.KDDiff(14))
metric_handler.add(metric.KDDiff(18))
metric_handler.add(metric.KDDiff(22))

metric_handler.add(metric.UltimateOscillator(7,14,28))
metric_handler.add(metric.UltimateOscillator(8,16,32))
metric_handler.add(metric.UltimateOscillator(9,18,36))

metric_handler.add(metric.MoneyFlowIndex(14))
metric_handler.add(metric.MoneyFlowIndex(18))
metric_handler.add(metric.MoneyFlowIndex(22))

label_handler = label.Label()

res = []
labels = []
start_date = datetime.datetime.strptime('01-10-2016', DATE_FORMAT)
end_date = datetime.datetime.strptime('01-11-2016', DATE_FORMAT)
while start_date != end_date:
    date = start_date

    one_day_data = finance_io.get_one_day_data('spy', date=date)
    if one_day_data is not None:
        row = dict()
        row['date'] = one_day_data['Date'].values[0]
        row['data'] = one_day_data['Close'].values[0]
        r = metric_handler.feed(row=row)
        res.append(r)

        #get label
        l = label_handler.get_label_for('spy', date, finance_io, take_difference=True, current_close=row['data'])
        labels.append(l)

    start_date += datetime.timedelta(days=1)

print()



