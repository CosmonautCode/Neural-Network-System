# Apple Stock Price Examples for Testing

Based on real historical Apple stock data, here are some example inputs you can try:

## Recent Examples (2020-2021 Era - High Prices)

### Example 1: Recent Trading Day
```
Open: 147.03
High: 148.57
Low: 144.90
Close: 146.14
Volume: 71598400
```
*Expected Adj Close: ~$146.14*

### Example 2: Volatile Day
```
Open: 144.35
High: 149.00
Low: 144.10
Close: 148.71
Volume: 67808200
```
*Expected Adj Close: ~$148.71*

### Example 3: Down Day
```
Open: 147.08
High: 147.95
Low: 142.53
Close: 142.64
Volume: 69473000
```
*Expected Adj Close: ~$142.64*

## Historical Examples

### 2015 Era (Medium Prices)
```
Open: 112.50
High: 115.20
Low: 111.80
Close: 114.30
Volume: 45000000
```
*Expected Adj Close: ~$114-115*

### 2010 Era (Lower Prices)
```
Open: 45.20
High: 46.80
Low: 44.90
Close: 46.30
Volume: 120000000
```
*Expected Adj Close: ~$46-47*

### Early 2000s (Very Low Prices)
```
Open: 8.50
High: 9.20
Low: 8.30
Close: 9.10
Volume: 80000000
```
*Expected Adj Close: ~$9-10*

## Price Ranges to Expect

**Historical Range (1980-2021):**
- **Minimum:** ~$0.04 (adjusted close)
- **Maximum:** ~$182.13 (adjusted close)
- **Typical Range:** $50-180 (recent years)

**Volume Ranges:**
- **Low Volume:** 10-50 million shares
- **Normal Volume:** 50-100 million shares  
- **High Volume:** 100-200+ million shares

## Tips for Testing

1. **Realistic Relationships:** High should be ≥ Open, Close, and Low
2. **Reasonable Spreads:** High-Low difference typically 1-5% of price
3. **Volume Patterns:** Higher volume on big price moves
4. **Try Different Eras:** Test with both low and high price ranges

## Example Test Cases to Try

### Conservative Test
```
Open: 150.00
High: 152.00
Low: 148.50
Close: 151.00
Volume: 75000000
```

### Volatile Test
```
Open: 145.00
High: 155.00
Low: 143.00
Close: 152.00
Volume: 120000000
```

### Low Price Era Test
```
Open: 12.50
High: 13.20
Low: 12.30
Close: 13.00
Volume: 95000000
```

Remember: This is for educational purposes only. The model was trained on historical data and may not predict future prices accurately!
