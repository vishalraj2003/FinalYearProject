# Generated by Django 5.0.1 on 2024-03-02 14:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('credit_balance_update', '0002_rename_timestamp_credit_balance_update_transaction_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='credit_balance_update',
            name='transaction_type',
            field=models.CharField(max_length=10, null=True),
        ),
    ]