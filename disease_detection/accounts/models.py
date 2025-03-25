from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class DetectionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='detection_history/')
    prediction = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']  # Most recent first

    def __str__(self):
        return f"{self.user.username} - {self.prediction} - {self.timestamp}"
