import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(42)

#################################
# Parameters
#################################

S0=100
K=100
r=0.05
mu=0.15          # upward drift
sigma=0.5        # high volatility
T=1

N=1000
M=1000

dt=T/N
times=np.linspace(0,T,N+1)

# Black-Scholes

def call_price(S,t):

    tau=T-t

    if tau<=0:
        return max(S-K,0)

    d1=(np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    d2=d1-sigma*np.sqrt(tau)

    return (
        S*norm.cdf(d1)
        -K*np.exp(-r*tau)*norm.cdf(d2)
    )

def delta(S,t):

    tau=T-t

    if tau<=0:
        return 1 if S>K else 0

    d1=(np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))

    return norm.cdf(d1)


# One path demonstration

while True:

    stock=[S0]
    S=S0

    for i in range(N):

        Z=np.random.randn()

        S*=np.exp(
            (mu-0.5*sigma**2)*dt
            +sigma*np.sqrt(dt)*Z
        )

        stock.append(S)

    # keep only upward paths
    if stock[-1]>140:
        break


stock=np.array(stock)

# Frequent hedge

cash=call_price(S0,0)
d=0

freq_pnl=[]

for i,S in enumerate(stock):

    t=i*dt

    portfolio=cash+d*S

    short_call_pnl=call_price(S0,0)-call_price(S,t)

    freq_pnl.append(
        short_call_pnl + portfolio
    )

    if i==N:
        break

    target=delta(S,t)

    extra=target-d

    cash-=extra*S

    d=target


# Sparse hedge

cash=call_price(S0,0)
d=0

sparse_pnl=[]

for i,S in enumerate(stock):

    t=i*dt

    portfolio=cash+d*S

    short_call_pnl=call_price(S0,0)-call_price(S,t)

    sparse_pnl.append(
        short_call_pnl + portfolio
    )

    if i==N:
        break

    if i%40==0:

        target=delta(S,t)

        extra=target-d

        cash-=extra*S

        d=target


# Monte Carlo losses

freq_losses=[]
sparse_losses=[]

for sim in range(M):

    S=S0

    stock=[S0]

    for i in range(N):

        Z=np.random.randn()

        S*=np.exp(
            (mu-0.5*sigma**2)*dt
            +sigma*np.sqrt(dt)*Z
        )

        stock.append(S)

    stock=np.array(stock)


    cash=call_price(S0,0)
    d=0

    for i,S in enumerate(stock[:-1]):

        t=i*dt

        target=delta(S,t)

        extra=target-d

        cash-=extra*S

        d=target

    portfolio=cash+d*stock[-1]

    payoff=max(stock[-1]-K,0)

    freq_losses.append(
        payoff-portfolio
    )


    cash=call_price(S0,0)
    d=0

    for i,S in enumerate(stock[:-1]):

        if i%40==0:

            t=i*dt

            target=delta(S,t)

            extra=target-d

            cash-=extra*S

            d=target

    portfolio=cash+d*stock[-1]

    payoff=max(stock[-1]-K,0)

    sparse_losses.append(
        payoff-portfolio
    )


# Plot 1: stock rises

plt.figure()
plt.plot(times,stock)
plt.title("Stock Path (Rising Scenario)")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()


# Plot 2: Hedged P&L

plt.figure()

plt.plot(
    times,
    freq_pnl,
    label="Frequent hedge"
)

plt.plot(
    times,
    sparse_pnl,
    label="Sparse hedge"
)

plt.axhline(0)

plt.legend()

plt.title(
    "Short Call + Hedge P&L"
)

plt.xlabel("Time")
plt.ylabel("P&L")

plt.show()


# Plot 3: Histogram

plt.figure(figsize=(8,8))

plt.subplot(2,1,1)
plt.hist(freq_losses,bins=40)
plt.title(
    "Frequent Hedge Final Loss"
)

plt.subplot(2,1,2)
plt.hist(sparse_losses,bins=40)
plt.title(
    "Sparse Hedge Final Loss"
)

plt.tight_layout()
plt.show()


print(
"Frequent std:",
np.std(freq_losses)
)

print(
"Sparse std:",
np.std(sparse_losses)
)