import random
from math import ceil
from decimal import Decimal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pandas import date_range
from datetime import timedelta
import joblib
from django.db.models import Count
from django.shortcuts import render, redirect
import requests
import time
from itertools import chain
from operator import attrgetter
import psutil
import os
from django.contrib.auth.hashers import make_password
from datetime import datetime
from django.contrib.auth.hashers import check_password

from django.template.loader import render_to_string

from analysis_request.models import Monitor
from urllib.parse import urlparse
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
import pandas as pd
import requests
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render, get_object_or_404
from django.db.models import Q
from django.utils import timezone
from django.core.mail import send_mail
from django.http import JsonResponse
from authentication_module import models
from authentication_module.models import signupModel
from django.contrib import messages
from django.shortcuts import redirect, render, reverse
import re
from about_us.models import about_us_model
from django.utils.dateparse import parse_date
from yahoo_fin import stock_info as si
from companyData.models import companyData
import yfinance as yf
from yahoo_fin import stock_info as si
from django.http import HttpResponseRedirect
from datetime import datetime
import warnings
# from authentication_module.models import signupModel
from django.core.paginator import Paginator
from functools import wraps
from django.db import transaction as db_transaction

from djangoProject import settings
from transaction.models import *
# from authentication_module.models import signupModel
from django.db.models import Sum
from django.db.models import F, Sum, ExpressionWrapper, DecimalField, Case, When
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from updateTracker.models import UpdateTracker
from authentication_module.models import signupModel
from credit_balance_update.models import credit_balance_update


# from django.http import HttpResponse


# Create your views here.
# context = {}
def transaction_history(request):
    user_email = request.COOKIES.get('email')
    user2 = signupModel.objects.filter(email=user_email).all()
    user = signupModel.objects.get(email=user_email)
    user_credit_balancef = float(user.credit_balance)
    amount = float(request.POST.get('amount', 0))
    current_time = datetime.now()

    user1 = credit_balance_update.objects.filter(email=user_email).all().order_by('-transaction_date')
    user_buy1 = BuyTransaction.objects.filter(user=user_email, transaction_type='buy').all().order_by(
        '-transaction_date')
    user_buy = BuyTransaction.objects.filter(user=user_email).first()
    date_buy = user_buy.transaction_date if user_buy else None
    transaction_type_buy = user_buy.transaction_type if user_buy else None
    buy_price = user_buy.buy_price_per_unit if user_buy else None
    quantity = user_buy.quantity if user_buy else None
    total_price_buy = buy_price * quantity if user_buy else None

    user_sell1 = SellTransaction.objects.filter(user=user_email, transaction_type='sell').all().order_by(
        '-transaction_date')
    user_sell = SellTransaction.objects.filter(user=user_email).first()
    date_sell = user_sell.transaction_date if user_sell else None
    transaction_type_sell = user_sell.transaction_type if user_sell else None
    sell_price = user_sell.sell_price_per_unit if user_sell else None
    quantity = user_sell.quantity if user_sell else None
    total_price_sell = sell_price * quantity if user_sell else None
    # user1 = credit_balance_update.objects

    combined_transactions = list(chain(user_buy1, user_sell1))
    sorted_transactions = sorted(combined_transactions, key=attrgetter('transaction_date'), reverse=True)

    all_transactions = sorted_transactions

    paginator = Paginator(all_transactions, 5)

    page_number = request.GET.get('page')
    all_transactions_final = paginator.get_page(page_number)

    try:
        transactions_paginated = paginator.page(page_number)
    except PageNotAnInteger:
        transactions_paginated = paginator.page(1)
    except EmptyPage:
        transactions_paginated = paginator.page(paginator.num_pages)

    if request.method == 'POST':
        user1 = credit_balance_update.objects.all()
        user1.email = user_email
        user_credit_balance1 = float(user.credit_balance)
        user1.previous_balance = user_credit_balance1
        user_credit_balancef = amount + user_credit_balance1
        user.credit_balance = user_credit_balancef
        user1.current_balance = user_credit_balancef
        user1.transaction_date = current_time

        # time1 = user1.timestamp
        credit_balance_update.objects.create(
            email=user_email,
            previous_balance=user_credit_balance1,
            current_balance=user_credit_balancef,
            transaction_type='credit',
            transaction_date=current_time
        )

        user.save()
        messages.success(request, 'Balance added successfully!')
        return redirect('payment')

    context = {
        'user_credit_balancef': user_credit_balancef,
        'email': user_email,
        'amount': amount if request.method == 'POST' else 0,
        'date_buy': date_buy,
        'date_sell': date_sell,
        'total_price_buy': total_price_buy,
        'total_price_sell': total_price_sell,
        'transaction_type_sell': transaction_type_sell,
        'transaction_type_buy': transaction_type_buy,
        'user_buy': user_buy1,
        # 'user_sell':user_sell,
        'user_sell': user_sell1,
        'time': current_time,
        'user1': user1,
        'transactions_paginated': transactions_paginated,

    }
    # redirect_url_1 = 'payment.html'
    # redirect_url_2 = 'transaction_history.html'
    return render(request,'transaction_history.html', context)

def payment(request):
    user_email = request.COOKIES.get('email')
    user2 = signupModel.objects.filter(email=user_email).all()
    user = signupModel.objects.get(email=user_email)
    user_credit_balancef = float(user.credit_balance)
    amount = float(request.POST.get('amount', 0))
    current_time = datetime.now()

    user1 = credit_balance_update.objects.filter(email=user_email).all().order_by('-transaction_date')
    user_buy1 = BuyTransaction.objects.filter(user=user_email, transaction_type='buy').all().order_by(
        '-transaction_date')
    user_buy = BuyTransaction.objects.filter(user=user_email).first()
    date_buy = user_buy.transaction_date if user_buy else None
    transaction_type_buy = user_buy.transaction_type if user_buy else None
    buy_price = user_buy.buy_price_per_unit if user_buy else None
    quantity = user_buy.quantity if user_buy else None
    total_price_buy = buy_price * quantity if user_buy else None

    user_sell1 = SellTransaction.objects.filter(user=user_email, transaction_type='sell').all().order_by(
        '-transaction_date')
    user_sell = SellTransaction.objects.filter(user=user_email).first()
    date_sell = user_sell.transaction_date if user_sell else None
    transaction_type_sell = user_sell.transaction_type if user_sell else None
    sell_price = user_sell.sell_price_per_unit if user_sell else None
    quantity = user_sell.quantity if user_sell else None
    total_price_sell = sell_price * quantity if user_sell else None
    # user1 = credit_balance_update.objects

    combined_transactions = list(chain(user1))
    sorted_transactions = sorted(combined_transactions, key=attrgetter('transaction_date'), reverse=True)

    all_transactions = sorted_transactions

    paginator = Paginator(all_transactions, 5)

    page_number = request.GET.get('page')
    all_transactions_final = paginator.get_page(page_number)

    try:
        transactions_paginated = paginator.page(page_number)
    except PageNotAnInteger:
        transactions_paginated = paginator.page(1)
    except EmptyPage:
        transactions_paginated = paginator.page(paginator.num_pages)

    if request.method == 'POST':
        user1 = credit_balance_update.objects.all()
        user1.email = user_email
        user_credit_balance1 = float(user.credit_balance)
        user1.previous_balance = user_credit_balance1
        user_credit_balancef = amount + user_credit_balance1
        user.credit_balance = user_credit_balancef
        user1.current_balance = user_credit_balancef
        user1.transaction_date = current_time

        # time1 = user1.timestamp
        credit_balance_update.objects.create(
            email=user_email,
            previous_balance=user_credit_balance1,
            current_balance=user_credit_balancef,
            transaction_type='credit',
            transaction_date=current_time
        )

        user.save()
        messages.success(request, 'Balance added successfully!')
        return redirect('payment')

    context = {
        'user_credit_balancef': user_credit_balancef,
        'email': user_email,
        'amount': amount if request.method == 'POST' else 0,
        'date_buy': date_buy,
        'date_sell': date_sell,
        'total_price_buy': total_price_buy,
        'total_price_sell': total_price_sell,
        'transaction_type_sell': transaction_type_sell,
        'transaction_type_buy': transaction_type_buy,
        'user_buy': user_buy1,
        # 'user_sell':user_sell,
        'user_sell': user_sell1,
        'time': current_time,
        'user1': user1,
        'transactions_paginated': transactions_paginated,

    }

    return render(request, 'payment.html', context)


def stock_details(request, company_symbol):
    # Fetch transactions and company data
    buy_transactions = BuyTransaction.objects.filter(company_symbol=company_symbol).values()
    sell_transactions = SellTransaction.objects.filter(company_symbol=company_symbol).values()
    company = get_object_or_404(companyData, symbol=company_symbol)
    current_price = Decimal(company.quote_price)

    # Convert to DataFrame
    transactions = list(buy_transactions) + list(sell_transactions)
    df = pd.DataFrame(list(transactions))

    if not df.empty:
        # Ensure 'transaction_date' is a datetime type
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])

        # Ensure 'quantity' and 'price_per_unit' are numeric
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['price_per_unit'] = pd.to_numeric(df.get('buy_price_per_unit', 0), errors='coerce', downcast='float')
        df['sell_price_per_unit'] = pd.to_numeric(df.get('sell_price_per_unit', 0), errors='coerce', downcast='float')
        df['current_price'] = current_price

        # Calculate profit for each transaction
        df['profit'] = df.apply(
            lambda row: (Decimal(row['current_price']) - Decimal(row['price_per_unit'])) * Decimal(row['quantity']) if
            row['transaction_type'] == 'buy' else (Decimal(row['sell_price_per_unit']) - Decimal(
                row['price_per_unit'])) * Decimal(row['quantity']), axis=1)

        # Separate dataframes for buy and sell transactions
        # df_buy = df[df['transaction_type'] == 'buy']
        # df_sell = df[df['transaction_type'] == 'sell']

        # Plotting
        plt.figure(figsize=(10, 6))

        df_buy = df[df['transaction_type'] == 'buy']
        if not df_buy.empty:
            plt.plot(df_buy['transaction_date'], df_buy['profit'], label='Buy Profit', marker='o', linestyle='-',
                     color='blue')

        # Plot Sell Transactions Profit
        df_sell = df[df['transaction_type'] == 'sell']
        if not df_sell.empty:
            plt.plot(df_sell['transaction_date'], df_sell['profit'], label='Sell Profit', marker='x', linestyle='-',
                     color='orange')

        plt.title(f'Profit from Transactions for {company.companyName}')
        plt.xlabel('Date')
        plt.ylabel('Profit')
        plt.legend()
        buf_profit = BytesIO()
        plt.savefig(buf_profit, format='png', bbox_inches='tight')
        plt.close()
        profit_image_base64 = base64.b64encode(buf_profit.getvalue()).decode('utf-8')
        buf_profit.close()

        # quantity graph
        plt.figure(figsize=(10, 6))
        if not df_buy.empty:
            plt.plot(df_buy['transaction_date'], df_buy['quantity'], label='Buy Quantity', marker='o', linestyle='-',
                     color='green')
        if not df_sell.empty:
            plt.plot(df_sell['transaction_date'], df_sell['quantity'], label='Sell Quantity', marker='x', linestyle='-',
                     color='red')
        plt.title(f'Buy and Sell Quantities for {company.companyName}')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        buf_quantity = BytesIO()
        plt.savefig(buf_quantity, format='png', bbox_inches='tight')
        plt.close()
        quantity_image_base64 = base64.b64encode(buf_quantity.getvalue()).decode('utf-8')
        buf_quantity.close()
    else:
        profit_image_base64 = None
        quantity_image_base64 = None

    # Buy Price vs Date Graph
    buy_price_image_base64 = generate_price_vs_date_graph(buy_transactions, 'buy_price_per_unit',
                                                          'Buy Price Per Unit vs Date', 'blue')

    # Sell Price vs Date Graph
    sell_price_image_base64 = generate_price_vs_date_graph(sell_transactions, 'sell_price_per_unit',
                                                           'Sell Price Per Unit vs Date', 'orange')
    context = {
        'company_symbol': company_symbol,
        'profit_image_base64': profit_image_base64,
        'quantity_image_base64': quantity_image_base64,
        'buy_price_image_base64': buy_price_image_base64,
        'sell_price_image_base64': sell_price_image_base64,
    }

    return render(request, 'stock_details.html', context)


def generate_price_vs_date_graph(transactions, price_column, title, color):
    if transactions:
        df = pd.DataFrame(transactions)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        plt.figure(figsize=(10, 6))
        plt.plot(df['transaction_date'], df[price_column], label=title, marker='o', linestyle='-', color=color)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price Per Unit')
        plt.legend()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    else:
        return None


def portfolioView(request):
    search_query = request.GET.get('search', '').strip()
    user_email = request.COOKIES.get('email')
    if not user_email:
        # Handle user not logged in scenario
        return render(request, 'login.html')
    if user_email:
        transactions = Transaction.objects.filter(user=user_email).values(
            'company_symbol', 'transaction_type'
        ).annotate(
            total_quantity=Sum('quantity'),
            total_buy_amount=Sum(
                Case(
                    When(transaction_type='buy',
                         then=ExpressionWrapper(F('quantity') * F('buy_price_per_unit'), output_field=DecimalField())),
                    default=0,
                    output_field=DecimalField()
                )
            ),
            total_sell_amount=Sum(
                Case(
                    When(transaction_type='sell',
                         then=ExpressionWrapper(F('quantity') * F('sell_price_per_unit'), output_field=DecimalField())),
                    default=0,
                    output_field=DecimalField()
                )
            )
        ).order_by('company_symbol', 'transaction_type')
        total_profit_loss = 0
        total_sell_profit_loss_port = Decimal(0)
        portfolio = []
        for trans in transactions:
            company = companyData.objects.get(symbol=trans['company_symbol'])
            # Calculate profit or loss
            current_price = Decimal(company.quote_price)  # Assuming this is the current stock price

            # total_investment = trans['total_amount']
            if trans['transaction_type'] == 'buy':
                total_investment = trans['total_buy_amount']
            elif trans['transaction_type'] == 'sell':
                total_investment = trans['total_sell_amount']
            else:
                total_investment = Decimal(0)

            market_value = trans['total_quantity'] * current_price
            profit_loss = Decimal(market_value) - Decimal(total_investment)
            if trans['transaction_type'] == 'buy':
                total_profit_loss += profit_loss

            sell_profit_loss = 0

            if trans['transaction_type'] == 'sell':
                sell_transactions = SellTransaction.objects.filter(
                    user=user_email,
                    company_symbol=trans['company_symbol']
                ).aggregate(total_sell_profit_loss=Sum('sell_profit_loss'))
                sell_profit_loss = sell_transactions['total_sell_profit_loss']
                total_sell_profit_loss_port += sell_profit_loss

            portfolio.append({
                'company_name': company.companyName,
                'symbol': trans['company_symbol'],
                'quantity': trans['total_quantity'],
                'transaction_type': trans['transaction_type'],
                'total_investment': total_investment,
                'current_price': current_price,
                'market_value': market_value,
                'buy_profit_loss': profit_loss,
                'sell_profit_loss': sell_profit_loss,
                'profit_loss_color': 'green' if profit_loss > 0 else 'red',
                'profit_loss_color2': 'green' if sell_profit_loss > 0 else 'red',
            })

        if search_query:
            filtered_portfolio = [item for item in portfolio if
                                  search_query.lower() in item['company_name'].lower() or search_query.lower() in item[
                                      'symbol'].lower()]
        else:
            filtered_portfolio = portfolio
        stocks_per_page = 3  # or any number you prefer
        page_number = request.GET.get('page')  # Get the page number from request
        paginator_stocks = Paginator(filtered_portfolio, stocks_per_page)
        try:
            stocks_page_obj = paginator_stocks.get_page(page_number)
        except PageNotAnInteger:
            stocks_page_obj = paginator_stocks.page(1)
        except EmptyPage:
            stocks_page_obj = paginator_stocks.page(paginator_stocks.num_pages)

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            html = render_to_string(
                template_name='portfolio_stocks_list.html',
                context={'portfolio': stocks_page_obj}
            )
            return JsonResponse({'html': html})

        context = {
            'portfolio': stocks_page_obj,
            'total_profit_loss': total_profit_loss,
            'total_sell_profit_loss': total_sell_profit_loss_port,

        }
        return render(request, 'portfolio.html', context)
    return render(request, 'portfolio.html')


def sell_stock(request, company_symbol):
    if request.method == "POST":
        user_email = request.COOKIES.get('email', '')
        quantity_to_sell = int(request.POST.get("quantity"))
        company = get_object_or_404(companyData, symbol=company_symbol)

        buy_transaction = Transaction.objects.filter(user=user_email, company_symbol=company_symbol,transaction_type='buy').first()
        total_bought = BuyTransaction.objects.filter(user=user_email, company_symbol=company_symbol, transaction_type='buy').aggregate(Sum('quantity'))['quantity__sum'] or 0
        total_sold = SellTransaction.objects.filter(user=user_email, company_symbol=company_symbol, transaction_type='sell').aggregate(Sum('quantity'))['quantity__sum'] or 0
        total_owned = total_bought - total_sold
        print("total_bought:",total_bought)
        print("total_sold",total_sold)
        print("total_owned",total_owned)
        print("quantity_to_sell",quantity_to_sell)
        print(buy_transaction)
        if True:

            transaction = Transaction(
                user=user_email,
                company_symbol=company_symbol,
                transaction_type='sell',
                quantity=quantity_to_sell,
                sell_price_per_unit=company.quote_price,
                transaction_date=timezone.now()
            )
            transaction.save()

            user_data = signupModel.objects.filter(email=user_email).first()

            buy_price = buy_transaction.buy_price_per_unit * quantity_to_sell
            sell_price = Decimal(company.quote_price) * quantity_to_sell
            user_data.credit_balance += sell_price
            user_data.save()
            profit_loss = sell_price - buy_price

            # Create SellTransaction
            sell_transaction = SellTransaction(
                user=user_email,
                company_symbol=company_symbol,
                transaction_type='sell',
                quantity=quantity_to_sell,
                sell_price_per_unit=company.quote_price,
                buy_price_per_unit=buy_transaction.buy_price_per_unit,
                sell_profit_loss=profit_loss,
                transaction_date=timezone.now()
            )
            sell_transaction.save()

            buy_transaction.delete()
        else:
            return JsonResponse({"message": "Not enough stock", "status": "success"})

        return JsonResponse({
            "message": "Stock sold successfully. Check Portfolio. Your balance is "+str(user_data.credit_balance),
            "status": "success",
            "new_balance": float(user_data.credit_balance)
        })
    else:
        return JsonResponse({"message": "Invalid request method.", "status": "error"})



def buy_stock(request, company_symbol):
    if request.method == "POST":
        company = get_object_or_404(companyData, symbol=company_symbol)
        user_email = request.COOKIES.get('email', '')
        quantity = int(request.POST.get("quantity"))
        price_per_unit = company.quote_price
        user_data = signupModel.objects.filter(email=user_email).first()
        total_price = Decimal(price_per_unit) * quantity


        if user_data.credit_balance >= total_price:
            # Create Transaction record
            Transaction.objects.create(
                user=user_email,
                company_symbol=company_symbol,
                transaction_type='buy',
                quantity=quantity,
                buy_price_per_unit=price_per_unit
            )

            BuyTransaction.objects.create(
                user=user_email,
                company_symbol=company_symbol,
                transaction_type='buy',
                quantity=quantity,
                buy_price_per_unit=price_per_unit,
            )

            user_data.credit_balance -= total_price
            user_data.save()
            updated_balance = user_data.credit_balance
            message = "Stock purchased successfully. Check Portfolio and balance left is "+str(updated_balance)
            return JsonResponse({"message": message , "status": "success"})
        else:
            return JsonResponse({"message": "Not sufficient balance", "status": "error"})

    return JsonResponse({"message": "Invalid request method.", "status": "error"})


def transaction_page(request):
    company_symbol = request.GET.get('company_symbol')
    print(company_symbol)
    print("Company Symbol Received:", company_symbol)  # Debug print
    company = get_object_or_404(companyData, symbol=company_symbol)
    user_email = request.COOKIES.get('email', '')
    user_inner = get_object_or_404(signupModel, email=user_email)

    context = {
        'company': company,
        'user': user_inner
    }

    return render(request, 'transaction.html', context)


def transaction_page_sell(request):
    company_symbol = request.GET.get('company_symbol')
    print(company_symbol)
    print("Company Symbol Received:", company_symbol)  # Debug print
    company = get_object_or_404(companyData, symbol=company_symbol)
    user_email = request.COOKIES.get('email', '')
    user_inner = get_object_or_404(signupModel, email=user_email)

    context = {
        'company': company,
        'user': user_inner
    }

    return render(request, 'transaction_sell.html', context)


def logout(request):
    response = HttpResponseRedirect(reverse('login'))
    response.delete_cookie('email')
    response.delete_cookie('first_name')
    response.delete_cookie('last_name')
    response.delete_cookie('image')
    response.delete_cookie('login_status')
    print("Logged out successfully")
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"  # HTTP 1.1
    response["Pragma"] = "no-cache"  # HTTP 1.0
    response["Expires"] = "0"  # Proxies
    return response


# login fix
def login_required_cookie(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        login_status = request.COOKIES.get('login_status', 'False')
        if login_status != 'True':
            return redirect('login')
        return view_func(request, *args, **kwargs)

    return _wrapped_view


def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data


def get_stock_data_model(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='max')
    return data


def prepare_data(df):
    df = df.reset_index()
    df = df.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)
    df['Close_10_days_avg'] = df['Close'].rolling(window=10).mean()
    df.dropna(inplace=True)
    return df


def split_data(df):
    X = df[['Close_10_days_avg']].values[:-7]
    y = df['Close'].shift(-7).values[:-7]
    return train_test_split(X, y, test_size=0.3, random_state=42)


def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def predict_and_recommend(model, df):
    last_date = df['Date'].iloc[-1]
    next_7_days_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    last_10_days_avg = df['Close'].rolling(window=10).mean().iloc[-7:]
    predictions = model.predict(np.array(last_10_days_avg).reshape(-1, 1))
    predictions_with_dates = list(zip(next_7_days_dates.strftime('%Y-%m-%d'), predictions))
    current_price = df['Close'].iloc[-1]
    avg_predicted_price = np.mean(predictions)
    recommendation = "Buy the stock." if avg_predicted_price > current_price else "Sell the stock."
    return predictions_with_dates, recommendation


def get_company_data(request, symbol):
    try:
        company = companyData.objects.get(symbol=symbol)
        df = get_stock_data_model(symbol)
        df_prepared = prepare_data(df)
        X_train, X_test, y_train, y_test = split_data(df_prepared)
        model = train_model(X_train, y_train)
        predictions_with_dates, recommendation = predict_and_recommend(model, df_prepared)
        print(predictions_with_dates)
        data = {
            'companyName': company.companyName,
            'symbol': company.symbol,
            'one_year_target_est': company.one_year_target_est,
            'fifty_two_week_range': company.fifty_two_week_range,
            'ask': company.ask,
            'avg_volume': company.avg_volume,
            'beta_5y_monthly': company.beta_5y_monthly,
            'bid': company.bid,
            'days_range': company.days_range,
            'eps_ttm': company.eps_ttm,
            'earnings_date': company.earnings_date,
            'ex_dividend_date': company.ex_dividend_date,
            'forward_dividend_yield': company.forward_dividend_yield,
            'market_cap': company.market_cap,
            'open_price': company.open_price,
            'pe_ratio_ttm': company.pe_ratio_ttm,
            'previous_close': company.previous_close,
            'quote_price': company.quote_price,
            'volume': company.volume,
            'description': company.description,
            'recommendation': recommendation,
            'predictions_with_dates': predictions_with_dates,
        }

        # for graph
        current_date = datetime.now()
        formatted_date = current_date.strftime('%Y-%m-%d')

        stock_data = get_stock_data(symbol, '2021-01-01', formatted_date)




        formatted_stock_data = {
            'date': list(stock_data.index.strftime('%Y-%m-%d')),
            'open': list(stock_data['Open']),
            'high': list(stock_data['High']),
            'low': list(stock_data['Low']),
            'close': list(stock_data['Close']),

        }

        data['stockData'] = formatted_stock_data

        return JsonResponse(data)
    except companyData.DoesNotExist:
        return JsonResponse({'error': 'Company not found'}, status=404)


def jump(request):
    request.session['context'] = {
        'email': request.COOKIES.get('email'),
        'first_name': request.COOKIES.get('firstName'),
        'last_name': request.COOKIES.get('lastName'),
        'image': request.COOKIES.get('image'),
        'login_status': request.COOKIES.get('login_status')
    }
    user_email = request.COOKIES.get('email')
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')

    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')

    response = requests.get(
        'http://api.ipstack.com/' + ip + '?access_key=0c2c9d3722da795f2c6a6087504c3959'
    )
    rawData = response.json()

    continent = rawData['continent_name']
    country = rawData['country_name']
    city = rawData['city']
    capital = rawData['location']['capital']
    now = datetime.now()
    datetimenow = now.strftime("%Y-%m-%d %H:%M:%S")
    dateonly = datetimenow.split(' ')[0]
    saveNow = Monitor(
        user=request.COOKIES.get('email'),
        continent=continent,
        country=country,
        capital=capital,
        city=city,
        datetime=dateonly,
        ip=ip
    )
    saveNow.save()
    return render(request, 'jump.html')


def jump_updated(request):
    response = HttpResponseRedirect(reverse('login'))
    response.delete_cookie('email')
    response.delete_cookie('first_name')
    response.delete_cookie('last_name')
    response.delete_cookie('image')
    response.delete_cookie('login_status')
    print("Logged out successfully")
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"  # HTTP 1.1
    response["Pragma"] = "no-cache"  # HTTP 1.0
    response["Expires"] = "0"  # Proxies

    return response


# The Django view


def listing(request):
    if not is_user_logged_in(request):
        return redirect('login')

    search_query = request.GET.get('search', '')

    if search_query:

        companies = companyData.objects.filter(
            Q(companyName__icontains=search_query) | Q(symbol__icontains=search_query)).values('companyName', 'symbol',
                                                                                               'quote_price')
    else:
        companies = companyData.objects.all().values('companyName', 'symbol', 'quote_price')

    print("test:", companies)

    for company in companies:
        company['user_owns'] = False
        print(request.COOKIES.get('email'), company['symbol'])
        company['user_owns'] = Transaction.objects.filter(
            user=request.COOKIES.get('email'),
            company_symbol=company['symbol'],
            transaction_type='buy'
        ).aggregate(total_quantity=Sum('quantity'))['total_quantity'] or 0
        print("value of owner:", company['user_owns'])

    paginator = Paginator(companies, 4)
    page_number = request.GET.get('page', 1)
    current_page = paginator.get_page(page_number)
    print(companyData)
    totalpage = current_page.paginator.num_pages
    search_query = request.GET.get('search', '')

    try:
        current_page_number = int(page_number)
    except ValueError:
        current_page_number = 1

    max_display_pages = 3
    half_window_size = max_display_pages // 2

    start_page = max(current_page_number - half_window_size, 1)
    end_page = min(totalpage, start_page + max_display_pages - 1)

    if end_page - start_page < max_display_pages - 1:
        start_page = max(end_page - max_display_pages + 1, 1)  # when we are at end
    # loginData = request.session.get('loginData')
    # print("login data is",loginData)
    page_range = range(start_page, end_page + 1)
    # print("value of owner (final):", company['user_owns'])
    data = {
        'companyData': current_page,
        'last_page': totalpage,
        'page_range': page_range,
        'search_query': search_query,
        'companyData2': companies,
    }

    # print(request.session.get('loginData'))
    return render(request, 'list-page.html', data)


def update_nifty50():
    warnings.filterwarnings("ignore", category=FutureWarning)

    nifty50_list = si.tickers_nifty50()

    for symbol in nifty50_list:
        try:
            data = si.get_quote_table(symbol)
            # print(data)

            ticker = yf.Ticker(symbol)
            info = ticker.info
            # company_instance = companyData()
            quote_price = str(round(float(data.get('Quote Price', 0)), 2))
            defaults = {
                'companyName': str(info['longName']),
                'one_year_target_est': data.get('1y Target Est', ''),
                'fifty_two_week_range': data.get('52 Week Range', ''),
                'ask': data.get('Ask', ''),
                'avg_volume': data.get('Avg. Volume', ''),
                'beta_5y_monthly': data.get('Beta (5Y Monthly)', ''),
                'bid': data.get('Bid', ''),
                'days_range': data.get("Day's Range", ''),
                'eps_ttm': data.get('EPS (TTM)', ''),
                'earnings_date': data.get('Earnings Date', ''),
                'ex_dividend_date': parse_date(data.get('Ex-Dividend Date', '')),
                'forward_dividend_yield': data.get('Forward Dividend & Yield', ''),
                'market_cap': data.get('Market Cap', ''),
                'open_price': data.get('Open', ''),
                'pe_ratio_ttm': data.get('PE Ratio (TTM)', ''),
                'previous_close': data.get('Previous Close', ''),
                'quote_price': quote_price,
                'volume': data.get('Volume', ''),
                'description': str(info['longBusinessSummary'])
            }

            obj, created = companyData.objects.update_or_create(
                symbol=symbol,
                defaults=defaults
            )

            if created:
                print(f"Created new record for {symbol}")
            else:
                print(f"Updated existing record for {symbol}")
        except IndexError:
            print(f"Table structure not as expected for symbol: {symbol}")
            pass
        except Exception as e:
            print(f"An error occurred while processing {symbol}: {e}")


# update_nifty50()


def login(request):
    if is_user_logged_in(request):
        return redirect('list')
    # 123
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        # hashed_password = make_password(password)
        try:
            user_inner = signupModel.objects.get(email=email)
            if check_password(password, user_inner.password):
                first_name = user_inner.first_name
                last_name = user_inner.last_name
                image = user_inner.profile_pic.url
                email = user_inner.email
                otp_a = random.randint(100000, 999999)
                request.session['login_data'] = {
                    'email': email,
                    'first_name': first_name,
                    'last_name': last_name,
                    'image': image,
                    'otp_a': otp_a
                }
                context = {
                    'login_data': request.session.get('login_data')
                }
                print("Checked your password value and its correct value")
                message = "Your OTP for Account opening is " + str(otp_a)
                print(message)
                send_mail(
                    'Activate your account',
                    message,
                    'vishalraj20@gnu.ac.in',
                    [email],
                    fail_silently=False
                )

                return render(request, 'OTP-verify-login.html', context)
            else:
                messages.error(request, 'Invalid Credentials')
                print("Check your password value but not correct value")
                return redirect('login')
        except ObjectDoesNotExist:
            messages.error(request, 'No account found with this email address.')
            return redirect('login')

    return render(request, 'login.html')


def otp_login_login(request):
    if is_user_logged_in(request):
        return redirect('list')

    login_data = request.session.get('login_data')
    otp_a = login_data.get('otp_a')
    otpWeb = request.POST.get('otpWeb')
    print("otp_a is " + str(otp_a))
    print("otpWeb is " + str(otpWeb))
    email = login_data.get('email')
    message = "Your OTP for Account login is " + str(otp_a)
    if request.method == 'GET':
        if str(otp_a) == str(otp_a):
            print("OTP Verified for login")
            send_mail(
                'Activate your account',
                message,
                'vishalraj20@gnu.ac.in',
                [email],
                fail_silently=False
            )
    if request.method == 'POST':
        if str(otp_a) == str(otpWeb):

            values = signupModel.objects.get(email=email)
            first_name = values.first_name
            last_name = values.last_name
            email = values.email
            image = values.profile_pic.url if values.profile_pic else 'media/profile_pics/profile.png'
            context = {
                'email': request.COOKIES.get('email'),
                'first_name': request.COOKIES.get('first_name'),
                'last_name': request.COOKIES.get('last_name'),
                'login_status': request.COOKIES.get('login_status'),
                'image': request.COOKIES.get('image'),
            }
            print("name: ", request.COOKIES.get('first_name'))
            print("lname: ", request.COOKIES.get('last_name'))
            print("lname: ", request.COOKIES.get('image'))
            response = render(request, 'jump.html')
            response.set_cookie('email', email)
            response.set_cookie('first_name', first_name)
            response.set_cookie('last_name', last_name)
            response.set_cookie('image', image)
            response.set_cookie('login_status', True)
            return response
        else:
            messages.error(request, 'Wrong OTP... <br> Try Again')
            return render(request, 'OTP-verify-login.html', login_data)
    return render(request, 'OTP-verify-login.html', login_data)


def user(request):
    if is_user_logged_in(request):
        return redirect('list')
    return render(request, 'header_user.html')


def is_user_logged_in(request):
    return request.COOKIES.get('login_status') == 'True'


def about_us(request):
    if is_user_logged_in(request):
        return redirect('list')
    about_data = about_us_model.objects.all()
    data = {
        'about_data': about_data
    }
    daily_update_nifty50()
    return render(request, 'about-us.html', data)


def should_run_update_nifty50():
    today = datetime.now().date()
    last_run_record, created = UpdateTracker.objects.get_or_create(last_run_date=today)
    if created:
        # Function has not run today
        return True
    else:
        # Function has already run today
        return False


def daily_update_nifty50():
    if should_run_update_nifty50():
        update_nifty50()


daily_update_nifty50()


# traffic monitor
def traffic_monitor(request):
    dataSaved = Monitor.objects.all().order_by('-datetime')
    # Getting loadover15 minutes
    load1, load5, load15 = psutil.getloadavg()
    cpu_usage = int((load15 / os.cpu_count()) * 100)
    ram_usage = int(psutil.virtual_memory()[2])
    p = Paginator(dataSaved, 100)

    totalSiteVisits = (p.count)

    pageNum = request.GET.get('page', 1)
    page1 = p.page(pageNum)

    a = Monitor.objects.order_by().values('ip').distinct()
    pp = Paginator(a, 10)
    # shows number of items in page
    unique = (pp.count)
    # update time
    now = datetime.now()

    data = {
        "now": now,
        "unique": unique,
        "totalSiteVisits": totalSiteVisits,
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "dataSaved": page1,
    }

    continent_distribution = Monitor.objects.values('continent').annotate(total=Count('continent')).order_by('-total')
    country_distribution = Monitor.objects.values('country').annotate(total=Count('country')).order_by('-total')
    city_distribution = Monitor.objects.values('city').annotate(total=Count('city')).order_by('-total')
    ip_distribution = Monitor.objects.values('ip').annotate(total=Count('ip')).order_by('-total')

    data.update({
        "continent_distribution": list(continent_distribution),
        "country_distribution": list(country_distribution),
        "city_distribution": list(city_distribution),
        "ip_distribution": list(ip_distribution),
    })
    return render(request, 'traffic_anlysis.html', data)


def profile_setting_user(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirmPassword')
        # profile_pic = request.POST.get('profile_pic')
        # profile_pic = "profile_pics/" + profile_pic
        context = {
            'email': request.COOKIES.get('email'),
            'first_name': request.COOKIES.get('first_name'),
            'last_name': request.COOKIES.get('last_name'),
            'login_status': request.COOKIES.get('login_status'),
            'image': request.COOKIES.get('image'),
        }
        if password is None:
            messages.error(request, "Password is required.")
            return render(request, 'profile_setting_user.html', context)

        if password != confirm_password:
            messages.error(request, 'Passwords do not match')
            return render(request, 'profile_setting_user.html', context)

        password_errors1 = []
        if len(password) < 8:
            password_errors1.append('Password must be at least 8 characters long.')

        if not re.search("[a-z]", password):
            password_errors1.append('Password must include lowercase letters.')

        if not re.search("[A-Z]", password):
            password_errors1.append('Password must include uppercase letters.')

        if not re.search("[0-9]", password):
            password_errors1.append('Password must include numbers.')

        if not re.search("[!@#$%^&*(),.?\":{}|<>]", password):
            password_errors1.append('Password must include special characters (!, @, #, $, etc.).')

        if password_errors1:
            for error in password_errors1:
                messages.error(request, error)
            return render(request, 'profile_setting_user.html', context)

        try:

            user_inner = signupModel.objects.get(email=email)
            user_inner.first_name = first_name
            user_inner.last_name = last_name
            user_inner.email = email
            user_inner.password = make_password(password)
            if 'profile_pic' in request.FILES:
                user_inner.profile_pic = request.FILES['profile_pic']
            user_inner.save()
            messages.success(request, 'Your data have been updated... Please re-login')
            del user_inner
            response = HttpResponseRedirect(reverse('login'))
            response.delete_cookie('email')
            response.delete_cookie('first_name')
            response.delete_cookie('last_name')
            response.delete_cookie('image')
            response.delete_cookie('login_status')
            print("Logged out successfully")
            response["Cache-Control"] = "no-cache, no-store, must-revalidate"  # HTTP 1.1
            response["Pragma"] = "no-cache"  # HTTP 1.0
            response["Expires"] = "0"  # Proxies

            return response
            # return render(request, 'login.html')
        except User.DoesNotExist:
            messages.error(request, 'User does not exist')
        except Exception as e:
            messages.error(request, f'Error updating profile: {e}')
    else:
        email = request.COOKIES.get('email', '')
        first_name = request.COOKIES.get('first_name', '')
        last_name = request.COOKIES.get('last_name', '')
        image = request.COOKIES.get('image', '')

        context = {
            'email': email,
            'first_name': first_name,
            'last_name': last_name,
            'image': image,
        }
    return render(request, 'profile_setting_user.html', context)


def profile_setting(request):
    if is_user_logged_in(request):
        return redirect('list')
    # login_data = request.session.get('login_data')

    if request.method == 'POST':
        login_data = request.session.get('login_data')
        if not login_data:
            return render(request, 'profile_setting.html')
        email = request.POST.get('email')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirmPassword')
        # profile_pic = request.POST.get('profile_pic')
        # profile_pic = "profile_pics/" + profile_pic
        if password is None:
            messages.error(request, "Password is required.")
            return render(request, 'profile_setting.html', login_data)

        if password != confirm_password:
            messages.error(request, 'Passwords do not match')
            return render(request, 'profile_setting.html', login_data)

        password_errors1 = []
        if len(password) < 8:
            password_errors1.append('Password must be at least 8 characters long.')

        if not re.search("[a-z]", password):
            password_errors1.append('Password must include lowercase letters.')

        if not re.search("[A-Z]", password):
            password_errors1.append('Password must include uppercase letters.')

        if not re.search("[0-9]", password):
            password_errors1.append('Password must include numbers.')

        if not re.search("[!@#$%^&*(),.?\":{}|<>]", password):
            password_errors1.append('Password must include special characters (!, @, #, $, etc.).')

        if password_errors1:
            for error in password_errors1:
                messages.error(request, error)
            return render(request, 'profile_setting.html', login_data)

        try:

            user_inner = signupModel.objects.get(email=email)
            user_inner.first_name = first_name
            user_inner.last_name = last_name
            user_inner.email = email

            user_inner.password = make_password(password)
            # if 'profile_pic' in request.FILES:
            #     user_inner.profile_pic = request.FILES['profile_pic']
            user_inner.save()
            messages.success(request, 'Your data have been updated')
            del user_inner
            # return render(request, 'login.html')
            return redirect('login')
        except User.DoesNotExist:
            messages.error(request, 'User does not exist')
        except Exception as e:
            messages.error(request, f'Error updating profile: {e}')
    return render(request, 'profile_setting.html')


def reset(request):
    if is_user_logged_in(request):
        return redirect('list')
    if request.method == 'POST':

        email = request.POST.get('email')
        page = "login_reset"
        otp_a = random.randint(100000, 999999)
        context = {
            'email': email,
            'otp_a': otp_a
        }

        if signupModel.objects.filter(email=email).exists():
            print(f"OTP: {otp_a}")
            values = signupModel.objects.get(email=email)
            first_name = values.first_name
            last_name = values.last_name
            email = values.email
            image = values.profile_pic.url if values.profile_pic else 'media/profile_pics/profile.png'
            request.session['login_data'] = {

                'email': email,

                'otp_a': otp_a,
                'first_name': first_name,
                'last_name': last_name,
                'image': image
            }
            context = {
                'login_data': request.session.get('login_data')
            }
            message = "Your OTP for Account reset is " + str(otp_a)
            if str(otp_a) == str(otp_a):
                print("OTP Verified")
                send_mail(
                    'Activate your account',
                    message,
                    'vishalraj20@gnu.ac.in',
                    [email],
                    fail_silently=False
                )
                pass
            return render(request, 'OTP-verify-reset.html', context)
        else:
            messages.error(request, f'Entered email is not registered. Please try again. Email: {email}')
            return render(request, 'reset.html', context)

    return render(request, 'reset.html')


def otp_login(request):
    if is_user_logged_in(request):
        return redirect('list')
    login_data = request.session.get('login_data')
    email = login_data.get('email')
    otp_a = login_data.get('otp_a')
    page = "reset"
    if request.method == 'POST':
        otp_web = request.POST.get('otpWeb')
        print("I am outside the otp verify")
        if str(otp_a) == str(otp_web):
            values = signupModel.objects.get(email=email)
            print(values)

            return render(request, 'profile_setting.html', login_data)
        else:
            messages.error(request, '<b>Wrong OTP Code<br> Try Again...</b>')
            return render(request, 'OTP-verify-reset.html', login_data)

    return render(request, 'OTP-verify-reset.html')


def otp(request):
    if is_user_logged_in(request):
        return redirect('list')
    signup_data = request.session.get('signup_data')
    if not signup_data:
        return redirect('signup')

    firstName = signup_data.get('firstName')
    lastName = signup_data.get('lastName')
    email = signup_data.get('email')
    password = signup_data.get('password')
    otp_a = signup_data.get('otp_a')

    print(otp_a)
    message = "Your OTP for Account opening is " + str(otp_a)
    if request.method == 'GET':
        if otp_a:
            send_mail(
                'Activate your account',
                message,
                'vishalraj20@gnu.ac.in',
                [email],
                fail_silently=False
            )
            pass
            print("otp called from 'otp' while signup successful.")
    if request.method == 'POST':

        otp_web = request.POST.get('otpWeb')
        print("I am outside the otp verify")
        if str(otp_a) == str(otp_web):
            try:
                data = signupModel(first_name=firstName, last_name=lastName, email=email, password=password)
                data.save()
                messages.success(request, '<b>Sign Up Successful!!</b>')
                del request.session['signup_data']
                return render(request, 'signup.html', {'signup_data': signup_data})
            except Exception as e:
                messages.error(request, f"Sign Up failed {e}")
                return render(request, 'signup.html', {'signup_data': signup_data})
        else:
            messages.error(request, '<b>Wrong OTP Code<br> Try Again...</b>')
            return render(request, 'OTP-verify.html', {'signup_data': signup_data})

        # return render(request, 'OTP-verify.html', data)

    return render(request, 'OTP-verify.html', signup_data)


def signup(request):
    if is_user_logged_in(request):
        return redirect('list')
    if request.method == 'POST':
        firstName = request.POST.get('firstName')
        lastName = request.POST.get('lastName')
        email = request.POST.get('email')
        password = request.POST.get('password1')
        confirmPassword = request.POST.get('password2')
        data = {
            'firstName': firstName,
            'lastName': lastName,
            'email': email,
            'password': password,
            'confirmPassword': confirmPassword,
            'page': 'signup'

        }
        if firstName == '' or lastName == '' or email == '' or password == '' or confirmPassword == '':
            messages.error(request, 'Fill all Fields.')
            return render(request, 'signup.html')

        if password != confirmPassword:
            messages.error(request, 'Passwords do not match')
            return render(request, 'signup.html')

        if signupModel.objects.filter(email=email).exists():
            messages.error(request,
                           'Email already registered. <br> <b>please try sign in.</b> <br> use forget password')
            return render(request, 'signup.html')
        password_errors = []
        if len(password) < 8:
            password_errors.append('Password must be at least 8 characters long.')

        if not re.search("[a-z]", password):
            password_errors.append('Password must include lowercase letters.')

        if not re.search("[A-Z]", password):
            password_errors.append('Password must include uppercase letters.')

        if not re.search("[0-9]", password):
            password_errors.append('Password must include numbers.')

        if not re.search("[!@#$%^&*(),.?\":{}|<>]", password):
            password_errors.append('Password must include special characters (!, @, #, $, etc.).')

        if password_errors:
            for error in password_errors:
                messages.error(request, error)
            return render(request, 'signup.html')

        # Generating OTP for verification
        otp_a = random.randint(100000, 999999)
        hashed_password = make_password(password)
        request.session['signup_data'] = {
            'firstName': firstName,
            'lastName': lastName,
            'email': email,
            'password': hashed_password,
            'otp_a': otp_a
        }
        # context = {
        #     'signup_data': request.session['signup_data']
        # }
        return redirect('otp')

    return render(request, 'signup.html')
