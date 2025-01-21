<script>
import { chart } from "svelte-apexcharts";
import {Card, Select} from "flowbite-svelte";
import { Button, Modal, Label, Input, Checkbox } from 'flowbite-svelte';
let defaultModal = false;

let ticker_options = [
    {value: "NVDA", name: "NVIDIA Corporation (NVDA)"},
    {value: "INTC", name: "Intel Corporation (INTC)"},
    {value: "AAPL", name: "Apple Inc. (AAPL)"},
    {value: "NTDOY", name: "Nintendo Co., Ltd. (NTDOY)"},
    {value: "SPOT", name: "Spotify Technology S.A. (SPOT)"},
    {value: "ZM", name: "Zoom Communications Inc. (ZM)"}
]

function create_graph(name, color1, color2){
    return {
        chart: {
            type: "line", // Use "area" chart type to enable area under the line
            animations: {
                enabled: true,
                speed: 800,
                animateGradually: {
                    enabled: true,
                    delay: 150,
                },
                dynamicAnimation: {
                    enabled: true,
                    speed: 10,
                },
            },
            height: "400px", // Chart height
        },
        series: [
            {
                float_precision: 2,
                name: name,
                data: [],
            },
        ],
        stroke: {
            curve: "smooth", // Smooth line
            width: 2, // Line thickness
            colors: [color1],
            opacity: 0.3
        },
        fill: {
            type: "gradient", // Use gradient for the fill
            gradient: {
                shadeIntensity: 1,
                type: "vertical",
                gradientToColors: [color2],
                inverseColors: false,
                opacityFrom: 0.9, // Start opacity
                opacityTo: 0.9,
                stops: [0, 100],
            },
        },
        xaxis: {
            type: "datetime",
        },
        yaxis: {
            title: {
                text: name,
                style: {
                    fontSize: '16px', // Increased font size
                    fontWeight: 'bold',
                    color: '#333',
                },
            },
            type: "numeric",
            decimalsInFloat: 2,
        },
        colors: [color1],
        tooltip: {
            enabled: true,
        },
        dataLabels: {
            enabled: false,
        },
    };
}
let formModal = true;

let started = false;

// Initial chart options
let options_price = create_graph("Current stock price", "#723AF0", "#F03830");
let options_stock_history = create_graph("Stocks owned", "#F0B43A", "#86F030");
let options_balance = create_graph("Current balance", "#3AF050", "#309AF0")
let options_portfolio = create_graph("Current portfolio value", "#008FFB", "#FF4560")
const MAX_CHART_SIZE = 100;
let speed = 5
let ticker = "NVDA"
let baseBalance = 1000;

let current_price = 0
let balance = baseBalance
let holding = 0
let action = 0
let price_change = {
    increase: true,
    portfolio_value: baseBalance,
    change: 0,
    change_percent: 0
}

function start(event) {
    console.log(event.target)
    formModal = false;
    started = true;

    // WebSocket to fetch real-time data
    const ws = new WebSocket(`ws://localhost:8000/ws/${ticker}/${baseBalance}/${speed}`);
    balance = baseBalance;
    price_change.portfolio_value = baseBalance;

    ws.onmessage = (event) => {
        const newPoint = JSON.parse(event.data);
        current_price = newPoint.current_price;
        balance = newPoint.balance;
        holding = newPoint.stocks_owned;
        action = newPoint.action;

        price_change.portfolio_value = newPoint.stocks_owned * newPoint.current_price + newPoint.balance;
        price_change.increase = price_change.portfolio_value >= baseBalance;
        price_change.change = price_change.portfolio_value - baseBalance;
        price_change.change_percent = price_change.increase ? price_change.change / baseBalance * 100 - 100 : 100 - price_change.change / baseBalance * 100;

        // Update chart data dynamically
        options_price = {
            ...options_price,
            series: [
                {
                    float_precision: 2,
                    name: options_price.series[0].name,
                    data: [...options_price.series[0].data, [newPoint.timestamp, newPoint.current_price]],
                },
            ],
        };
        if (options_price.series[0].data.length > MAX_CHART_SIZE){
            options_price.series[0].data.shift()
        }
        options_stock_history = {
            ...options_stock_history,
            series: [
                {
                    float_precision: 2,
                    name: options_stock_history.series[0].name,
                    data: [...options_stock_history.series[0].data, [newPoint.timestamp, newPoint.stocks_owned]],
                },
            ],
        };
        if (options_stock_history.series[0].data.length > MAX_CHART_SIZE){
            options_stock_history.series[0].data.shift()
        }
        options_balance = {
            ...options_balance,
            series: [
                {
                    float_precision: 2,
                    name: options_balance.series[0].name,
                    data: [...options_balance.series[0].data, [newPoint.timestamp, newPoint.balance]],
                },
            ],
        };
        if (options_balance.series[0].data.length > MAX_CHART_SIZE){
            options_balance.series[0].data.shift()
        }
        options_portfolio = {
            ...options_portfolio,
            series: [
                {
                    float_precision: 2,
                    name: options_portfolio.series[0].name,
                    data: [...options_portfolio.series[0].data, [newPoint.timestamp, price_change.portfolio_value]],
                },
            ],
        };
        if (options_portfolio.series[0].data.length > MAX_CHART_SIZE){
            options_portfolio.series[0].data.shift()
        }
    };
}
</script>

<style>
	/* Chart styling */
	.chart-container {
		padding: 1rem;
		box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
		border-radius: 8px;
		background-color: #fff;
	}

    .grid {
        display: grid;
        grid-template-columns: 50% 50%;
        grid-template-rows: 50% 50%;
        gap: 10px;
    }

    .outer  {
        margin: 50px 15%;
        display: block;
    }

    .price-container {
        display: flex;
        align-items: center;
        align-self: center;
        align-content: center;
        font-family: "GT America", "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-size: 3rem;
        padding: 0;
        margin: 0;
    }

    .current-value {
        font-weight: bold;
        color: #333; /* Dark color for the main price */
    }

    .change {
        margin-left: 1rem;
        font-weight: bold;
        font-size: 2.5rem;
    }

    .change.positive {
        color: green; /* Green for positive change */
    }

    .change.negative {
        color: red; /* Red for negative change */
    }

    .outer-price {
        margin-bottom: 40px;
        margin-left: 40px;
        font-size: 3.5rem;
    }

    .outer-price h2 {
        margin-bottom: -10px;
        padding: 0;
    }
</style>

{#if started}
    <div class="outer">
        <div class="outer-price">
            <h2>Portfolio Value:</h2>
            <div class="price-container">

                <span class="current-value">{price_change.portfolio_value.toFixed(2)}</span>
                {#if price_change.increase}
                    <span class="change positive">+{(price_change.portfolio_value - baseBalance).toFixed(2)} ({((price_change.portfolio_value / (baseBalance) - 1) * 100).toFixed(2)}%)</span>
                {:else}
                    <span class="change negative">{(price_change.portfolio_value - baseBalance).toFixed(2)} ({-(100 - (price_change.portfolio_value * 100 / (baseBalance))).toFixed(2)}%)</span>
                {/if}
            </div>
        </div>
        <div class="grid">
            <div class="chart-container" use:chart={options_price}></div>
            <div class="chart-container" use:chart={options_stock_history}></div>
            <div class="chart-container" use:chart={options_balance}></div>
            <div class="chart-container" use:chart={options_portfolio}></div>
        </div>
    </div>
{/if}

<Modal bind:open={formModal} size="xs" autoclose={false} on:close={(e) => {console.log("started"); if (!started) location.reload()}} class="w-full" >
    <form class="flex flex-col space-y-6"  action="#" on:submit={start}>
        <h3 class="mb-4 text-xl font-medium text-gray-900 dark:text-white">Welcome!</h3>
        <Label class="space-y-2">
            <span>Initial balance</span>
            <Input type="number" name="start_balance" placeholder="1000.00" step="0.01" required bind:value={baseBalance}/>
        </Label>
        <Label class="space-y-2">
            <span>Stock selection</span>
            <Select items={ticker_options} bind:value={ticker} required />
        </Label>
        <Button type="submit" color="red" class="w-full1">Start trading now!</Button>
    </form>
</Modal>
