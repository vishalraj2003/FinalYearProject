import random
from math import ceil

import requests
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render
from django.db.models import Q

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

from django.core.paginator import Paginator
from functools import wraps


# from django.http import HttpResponse


# Create your views here.
# context = {}

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


def get_company_data(request, symbol):
    try:
        company = companyData.objects.get(symbol=symbol)
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
            'close': list(stock_data['Close'])
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

    # companies = companyData.objects.all().values('companyName', 'symbol', 'quote_price')
    paginator = Paginator(companies, 6)
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

    data = {
        'companyData': current_page,
        'last_page': totalpage,
        'page_range': page_range,
        'search_query': search_query
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
                'quote_price': data.get('Quote Price', ''),
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
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            user_inner = models.signupModel.objects.get(email=email)
            if user_inner.password == password:
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
    if request.method == 'POST':
        login_data = request.session.get('login_data')
        otp_a = login_data.get('otp_a')
        otpWeb = request.POST.get('otpWeb')
        print("otp_a is " + str(otp_a))
        print("otpWeb is " + str(otpWeb))
        if str(otp_a) == str(otpWeb):
            email = login_data.get('email')
            values = models.signupModel.objects.get(email=email)
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
    # update_nifty50()
    return render(request, 'about-us.html', data)


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

            user_inner = models.signupModel.objects.get(email=email)
            user_inner.first_name = first_name
            user_inner.last_name = last_name
            user_inner.email = email
            user_inner.password = password
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

            user_inner = models.signupModel.objects.get(email=email)
            user_inner.first_name = first_name
            user_inner.last_name = last_name
            user_inner.email = email
            user_inner.password = password
            if 'profile_pic' in request.FILES:
                user_inner.profile_pic = request.FILES['profile_pic']
            user_inner.save()
            messages.success(request, 'Your data have been updated')
            del user_inner
            return render(request, 'login.html')
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

        if models.signupModel.objects.filter(email=email).exists():
            print(f"OTP: {otp_a}")
            values = models.signupModel.objects.get(email=email)
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
            message = "Your OTP for Account opening is " + str(otp_a)
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
            values = models.signupModel.objects.get(email=email)
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
    if otp_a:
        send_mail(
            'Activate your account',
            message,
            'vishalraj20@gnu.ac.in',
            [email],
            fail_silently=False
        )
        pass
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

        if models.signupModel.objects.filter(email=email).exists():
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

        request.session['signup_data'] = {
            'firstName': firstName,
            'lastName': lastName,
            'email': email,
            'password': password,
            'otp_a': otp_a
        }
        # context = {
        #     'signup_data': request.session['signup_data']
        # }
        return redirect('otp')

    return render(request, 'signup.html')
