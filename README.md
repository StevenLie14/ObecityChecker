
# 🏨 Hotel Booking Cancellation Classification – UTS Deployment

## Student Info
- **Name:** Steven Liementha
- **NIM:** 2702265370
- **Class:** LA09
- **Video Link:** 
- [https://binusianorg-my.sharepoint.com/personal/steven_liementha_binus_edu/_layouts/15/guestaccess.aspx?share=EmMxAwP_XW5Ov2YazTS6ptgBujE0XQCz3bZYDUIV2q-N4w&e=qxGGwh](https://binusianorg-my.sharepoint.com/personal/steven_liementha_binus_edu/_layouts/15/guestaccess.aspx?share=EmMxAwP_XW5Ov2YazTS6ptgBujE0XQCz3bZYDUIV2q-N4w&e=qxGGwh) -> OneDrive
- [https://youtu.be/92B-bxRcdAw](https://youtu.be/92B-bxRcdAw) -> Youtube

### Streamlit Demo
🔗 **Live App:** [https://uts-model-deployment-c3al9wrtdffhybhstcmepr.streamlit.app](https://uts-model-deployment-c3al9wrtdffhybhstcmepr.streamlit.app)  
*Note: This link may expire or become inactive over time.*

## ⚙️ Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- **Python version:** `3.11.11` (recommended)

---

## Case Study Overview

As a Data Scientist, you are tasked with building and deploying a machine learning model to classify hotel booking cancellations. The model should predict whether a booking will be **Canceled** or **Not Canceled** based on the features provided in the dataset.

---

## Dataset Description

The dataset includes the following features:

| Column | Description |
|--------|-------------|
| `Booking_ID` | Unique identifier for each booking |
| `no_of_adults` | Number of adults |
| `no_of_children` | Number of children |
| `no_of_weekend_nights` | Number of weekend nights (Saturday or Sunday) stayed or booked |
| `no_of_week_nights` | Number of weekday nights (Monday to Friday) stayed or booked |
| `type_of_meal_plan` | Type of meal plan selected by the customer |
| `required_car_parking_space` | Whether parking space was required (0 = No, 1 = Yes) |
| `room_type_reserved` | Type of room reserved (values are encrypted by INN Hotels) |
| `lead_time` | Number of days between booking and arrival date |
| `arrival_year` | Year of arrival |
| `arrival_month` | Month of arrival |
| `arrival_date` | Date of arrival |
| `market_segment_type` | Designation of the market segment |
| `repeated_guest` | Whether the customer is a returning guest (0 = No, 1 = Yes) |
| `no_of_previous_cancellations` | Number of previous bookings that were canceled |
| `no_of_previous_bookings_not_canceled` | Number of previous bookings not canceled |
| `avg_price_per_room` | Average per-day price for the booking (in euros) |
| `no_of_special_requests` | Total number of special requests made by the customer (e.g., high floor, room with a view) |
| `booking_status` | **Target variable** – Indicates whether the booking was canceled or not |

### Create and Activate Conda Environment

```bash
conda create --name mlops python=3.11

conda activate mlops

pip install -r requirements.txt

python model.py # for training model and export model

python inference.py # for prediction

streamlit run streamlit.py #to run streamlit locally
```

### Note : To Update requirements.txt
```sh
pip list --format=freeze > requirements.txt
```
