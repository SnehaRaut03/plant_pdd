from django.contrib import admin
from .models import DetectionHistory

@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'prediction', 'timestamp')
    list_filter = ('user', 'prediction', 'timestamp')
    search_fields = ('user__username', 'prediction')
    ordering = ('-timestamp',)
