import random

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render
from django.core.mail import send_mail

from authentication_module import models
from authentication_module.models import signupModel
from django.contrib import messages
from django.shortcuts import redirect, render
import re
from about_us.models import about_us_model


# from django.http import HttpResponse


# Create your views here.
def login(request):
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
    login_data = request.session.get('login_data')
    otp_a = login_data.get('otp_a')
    otpWeb = request.POST.get('otpWeb')
    print("otp_a is " + str(otp_a))
    print("otpWeb is " + str(otpWeb))
    if str(otp_a) == str(otpWeb):
        return render(request, 'header_user.html', login_data)
    else:
        messages.error(request, 'Wrong OTP... <br> Try Again')
        return render(request, 'OTP-verify-login.html', login_data)



def user(request):
    return render(request, 'header_user.html')


def about_us(request):
    about_data = about_us_model.objects.all()
    data = {
        'about_data': about_data
    }
    return render(request, 'about-us.html', data)



def profile_setting(request):
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
