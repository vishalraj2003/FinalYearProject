from django.contrib import admin
from updateTracker.models import UpdateTracker


# Register your models here.
class updateTrackerAdmin(admin.ModelAdmin):
    list_display = (
        'last_run_date',
    )


admin.site.register(UpdateTracker, updateTrackerAdmin)
