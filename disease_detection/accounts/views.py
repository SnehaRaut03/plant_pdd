from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .forms import UserRegisterForm
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from detection.models import DetectionHistory
from django.utils import translation
from django.conf import settings

def signup(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'signup.html', {'form': form})

def is_admin(user):
    return user.userprofile.is_admin

@login_required
def profile(request):
    detection_count = DetectionHistory.objects.filter(user=request.user).count()
    return render(request, 'profile.html', {
        'detection_count': detection_count
    })

@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    users = User.objects.filter(userprofile__is_admin=False)
    return render(request, 'admin_dashboard.html', {'users': users})

@login_required
@user_passes_test(is_admin)
def user_details(request, user_id):
    user = User.objects.get(id=user_id)
    history = DetectionHistory.objects.filter(user=user)
    return render(request, 'user_details.html', {
        'viewed_user': user,
        'history': history
    })

@login_required
@user_passes_test(is_admin)
def manage_users(request):
    users = User.objects.filter(userprofile__is_admin=False).order_by('-date_joined')
    user_data = []
    total_detections = 0
    
    for user in users:
        detection_count = DetectionHistory.objects.filter(user=user).count()
        total_detections += detection_count
        user_data.append({
            'user': user,
            'detection_count': detection_count,
            'last_login': user.last_login,
            'date_joined': user.date_joined,
        })
    
    return render(request, 'manage_users.html', {
        'user_data': user_data,
        'total_detections': total_detections,
        'total_users': len(user_data)
    })

@login_required
@user_passes_test(is_admin)
def user_detail(request, user_id):
    viewed_user = get_object_or_404(User, id=user_id)
    history = DetectionHistory.objects.filter(user=viewed_user).order_by('-timestamp')
    
    return render(request, 'user_detail.html', {
        'viewed_user': viewed_user,
        'history': history
    })

def switch_language(request, language_code):
    """Direct language switching view"""
    response = redirect(request.GET.get('next', '/accounts/profile/'))
    translation.activate(language_code)
    response.set_cookie(settings.LANGUAGE_COOKIE_NAME, language_code)
    return response

