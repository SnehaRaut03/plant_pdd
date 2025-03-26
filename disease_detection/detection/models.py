from django.db import models
from django.contrib.auth.models import User

class DetectionHistory(models.Model):
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        related_name='plant_detection_history'
    )
    image = models.ImageField(
        upload_to='detection_images/%Y/%m/%d/',  # Organizes uploads by date
        null=True,
        blank=True
    )
    prediction = models.CharField(max_length=200)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']  # Most recent detections first
        verbose_name_plural = "Detection histories"

    def __str__(self):
        return f"{self.user.username} - {self.prediction} ({self.timestamp})"
