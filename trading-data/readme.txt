
Market_Data: 

Market data for France downloaded from the Transparency platform. Details:
	- Direction: if 1 then system=short, if 0 then system=long
	- Volume: if >0 then system is short, if <0 then system=long
	- Generation Forecast: Thermal generation capacity forecast (~ System Margin)		
	- Net Load Forecast = Load Forecast - Wind DA Forecast - Solar DA Forecast
	- Margin = Generation Forecast/ Net Load Forecast
	- Penalty = DA Price - Imbalance Price. Usually, if system = short then Penalty < 0. 
		   If system = long, then Penalty > 0
	- Pen_up/down: Upward/downward regulation costs. *Processed to be non-negative*
		If Volume > 0, then Pen_up=0 and Pen_down=Penalty
		If Volume < 0, then Pen_down=0 and Pen_up = Penalty

VPP_Data:

Renewable production data and NWP forecasts issued at 0:00am of day D-1 (24-48h horizon). The data concern an aggregation of 3 Wind Plants (WP)
and 1 Solar Plant (SP), located in Northern and Southern France.
- Norm_Power: normalized renewable production. Total capacity is 49MW.
- NWP forecasts: SSRD is solar radiation, Temp is temperature, WindSpeed is wind speed, WindDirection is wind direction forecasts.
	NWP forecasts for WPs are aggregated (average value).