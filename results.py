import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from db import close_db_pool, get_db_pool

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.facecolor'] = 'white'

codes = [
    "TCS80A107UL4",
    "TCS00Y3XYV94",
    "TCS00A106YF0",
    "TCS00A0ZZAC4",
    "RU000A106T36",
    "BBG00F9XX7H4",
    "BBG009GSYN76",
    "BBG008F2T3T2",
    "BBG004S68FR6",
    "BBG004S68BH6",
    "BBG004S68B31",
    "BBG004S689R0",
    "BBG004S68829",
    "BBG004S686W0",
    "BBG004S68614",
    "BBG004S685M3",
    "BBG004S68598",
    "BBG004S68507",
    "BBG004S68473",
    "BBG004S683W7",
    "BBG004S682Z6",
    "BBG004S681W1",
    "BBG004S681M2",
    "BBG004S681B4",
    "BBG004RVFFC0",
    "BBG00475KHX6",
    "BBG00475K6C3",
    "BBG00475K2X9",
    "BBG00475JZZ6",
    "BBG0047315Y7",
    "BBG0047315D0",
    "BBG004731489",
    "BBG004731354",
    "BBG004731032",
    "BBG004730ZJ9",
    "BBG004730RP0",
    "BBG004730N88",
    "BBG004730JJ5",
    "BBG000RMWQD4",
    "BBG000R607Y3",
    "BBG000QJW156",
]


async def fetch_data(code):
    db_pool = await get_db_pool()
    async with db_pool.acquire() as connection:
        rows = await connection.fetch("""
        SELECT date(sttm_indexes.to_time)   AS date,
       ROUND(sttm_indexes.index, 4) AS index,
        (SELECT s3.close_price
         FROM stocks AS s3
         WHERE s3.ts > sttm_indexes.to_time + INTERVAL '2 day'AND s3.ts<sttm_indexes.to_time + INTERVAL '7 day'
           AND s3.instrument_id = sttm_indexes.instrument_id
         ORDER BY s3.ts LIMIT 1) AS open_price1,
        (SELECT s4.close_price
         FROM stocks AS s4
         WHERE s4.ts > sttm_indexes.to_time + INTERVAL '2 day' AND s4.ts<sttm_indexes.to_time + INTERVAL '7 day'
           AND s4.instrument_id = sttm_indexes.instrument_id
         ORDER BY s4.close_price LIMIT 1) AS min,
       (SELECT s5.close_price
        FROM stocks AS s5
        WHERE s5.ts > sttm_indexes.to_time + INTERVAL '2 day' AND s5.ts<sttm_indexes.to_time + INTERVAL '7 day'
          AND s5.instrument_id = sttm_indexes.instrument_id
        ORDER BY s5.close_price DESC LIMIT 1) AS max

        FROM sttm_indexes
        WHERE sttm_indexes.instrument_id = $1
        ORDER BY sttm_indexes.to_time;


        """, code)
        return rows


def analyze_and_plot(df, code):
    df = df[df['Index'] != -1000000000]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df['Date'], df['Index'], label='STTM Index', color='#1f77b4', linestyle='None', marker='o')

    print(code)
    percent_up = 0
    count_up = 0
    percent_down = 0
    count_down = 0

    for i in range(len(df) - 1):
        index_val = df['Index'].iloc[i]
        price_now = df['Open Price'].iloc[i]
        if price_now is None:
            continue
        if index_val == 0:
            continue
        if index_val > 0:
            price_max = df['Max'].iloc[i]
            price_change = (price_max - price_now) / price_now
            if index_val > 1000:
                percent_up += price_change
                count_up += 1
        else:
            price_min = df['Min'].iloc[i]
            price_change = (price_now - price_min) / price_now
            if index_val < -1000:
                percent_down += price_change
                count_down += 1

        if abs(index_val) < 1000:
            color = 'grey'
        elif price_change > 0.02:
            color = 'green'
        elif price_change > 0 :
            color = 'yellow'
        elif price_change <= 0:
            color = 'red'

        ax1.axvspan(df['Date'].iloc[i], df['Date'].iloc[i + 1], color=color, alpha=0.1)

    print((percent_up * 100) / count_up)
    print((percent_down * 100) / count_down)
    ax1.set_title(
        f'{code}\nAverage profit: {((percent_up * 100) / count_up):.2f} '
        f'| Average loss: {((percent_down * 100) / count_down):.2f}')
    ax1.set_ylabel('STTM Index')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2.plot(df['Date'], df['Open Price'], label='Open Price', color='#ff7f0e', linewidth=2, marker='x')
    ax2.set_ylabel('Open Price')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'results/{code}_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return


async def main():
    for code in codes:
        data = await fetch_data(code)
        if len(data) == 0:
            print(f"No data for {code}")
            continue

        df = pd.DataFrame(data, columns=['Date', 'Index', 'Open Price', 'Min', 'Max'])
        df['Date'] = pd.to_datetime(df['Date'])
        analyze_and_plot(df, code)

    await close_db_pool()


if __name__ == "__main__":
    asyncio.run(main())
