"""
URL configuration for djangoProject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views. Home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from stockMarket import views
from django.conf import settings
from django.conf.urls.static import static
# importing views

# note: otp-verify-login-login for login otp
# otp-verify-login for reset otp
# otp-verify for sign up otp

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', views.login, name='login'),
    path('reset/', views.reset, name='reset'),
    path('otp-verify/', views.otp, name='otp'),
    path('otp-verify-login/', views.otp_login, name='otp-login'),
    path('otp-verify-login-login/', views.otp_login_login, name='otp-login-login'),
    path('signup/', views.signup, name='signup'),
    path('profile_setting/', views.profile_setting, name='profile_setting'),
    path('user/', views.user, name='user'),
    path('', views.about_us, name='about_us'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
