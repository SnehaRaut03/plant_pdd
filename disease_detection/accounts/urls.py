from django.urls import path
from django.contrib.auth import views as auth_views
from .views import signup, profile, admin_dashboard, user_details, manage_users, user_detail, switch_language
from . import views

urlpatterns = [
    path("signup/", signup, name="signup"),
    path("login/", auth_views.LoginView.as_view(template_name="login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(template_name="logout.html"), name="logout"),
    path('profile/', views.profile, name='profile'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('user-details/<int:user_id>/', views.user_details, name='user_details'),
    path('manage-users/', views.manage_users, name='manage_users'),
    path('user-detail/<int:user_id>/', views.user_detail, name='user_detail'),
    path('switch-language/<str:language_code>/', views.switch_language, name='switch_language'),
]