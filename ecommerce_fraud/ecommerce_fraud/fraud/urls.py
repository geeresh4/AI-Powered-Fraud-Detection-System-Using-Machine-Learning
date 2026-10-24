from django.contrib import admin
from django.urls import path
from fraudapp import views   # <-- your Django app name

urlpatterns = [
    path('admin/', admin.site.urls),

    # Home
    path('', views.home, name='home'),

    # Admin
    path('admin_login/', views.admin_login, name='admin_login'),
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),

    # Dataset Pipeline
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('preprocess/', views.preprocess_data, name='preprocess_data'),
    path('split/', views.split_data, name='split_data'),

    # ML Model Training
    path('train_xgboost/', views.train_xgboost, name='train_xgboost'),
    path('train_stacking/', views.train_stacking, name='train_stacking'),

    # User auth
    path('register/', views.register, name='register'),
    path('user_login/', views.user_login, name='user_login'),

    # User prediction
    path('user_input/', views.user_input, name='user_input'),

    # Logout
    path('logout/', views.logout, name='logout'),
]
