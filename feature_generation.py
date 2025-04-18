#
# Feature Extraction
#
import numpy as np

def map_price_level(x):
    """
    This function takes a ticket price level as input and maps it to a numerical value
    using a manually fixed projection of numeric values. If the input is a string and matches one of the predefined
    categories ('Adult', 'Youth', 'GA'), it returns the corresponding numerical value.
    If the input is numeric or does not match any predefined category, it returns the input as-is.
    If the input is invalid or cannot be converted, it returns NaN.

    Parameters:
        x (str or numeric): The ticket price level, which can be a string (e.g., 'Adult', 'Youth', 'GA') or a numeric value.

    Returns:
        float: The mapped numerical value of the ticket price level, or NaN if the input is invalid.
    """
    price_level_mapping = {
    'Adult': 5,
    'Youth': 6,
    'GA': 7
}
    if isinstance(x, str):
        return price_level_mapping.get(x, np.nan)
    else:
        return x

def feature_extraction(account_id, subscriptions_df,tickets_df,account_df):
    """
    Extracts features for specified account ID's from multiple datasets.

    This function extracts a broad feature sets for a given account ID 
    by aggregating data from subscription history, ticket purchase history, and various datapoints.
    The extracted features are used to capture behavior, preferences, and demographics to develop a model to generate 
    predictions on which patrons will subscribe for the 2014-2015 concert year.
    Please note: Not all provided data is fed into the model.

    Parameters:
        account_id: The unique identifier for the account.
        subscriptions_df: DataFrame containing subscription details.
        tickets_df: DataFrame containing ticket purchase details.
        account_df: DataFrame containing account-level information

    Returns:
        features: A dictionary containing the extracted features for a given account ID.
    """
        
    # Initialize features dictionary
    features = {
        'account_id': account_id,
        'has_subscription_history': 0,
        'total_tickets_purchased': 0,
        'subs_tier': 0,
        'seasons_attended': 0,
        'sub_price_level':0,
        'concert_variety':0,
        'composer_diversity': 0,
        'most_frequent_location': None,
        'preferred_concert_location': None,
        'total_subscriptions': 0,
        'average_price_level': 0,
        'median_price_level': 0,
        'sub_seat_number': 0,
        'sub_season': None,
        'most_frequent_set': None,
        'sub_location': None,
        'sub_section': None,
        'sub_package': None,
        'mult_subs': None,
        'billing_city': None,
        'first_donated': None,
        'initial_donation': 0,
        'average_seats_per_ticket': 0,
        'donation_history': 0,
        'donation_amount': 0,
        'donation_amount_2013': 0,
        'shipping_zip': 0,
        'total_tickets': 0,
        'billing_zip':0,
        'frequency': 0,
        

    }

    # Subscription history
    subscription_history = subscriptions_df[subscriptions_df['account.id'] == account_id]
    if not subscription_history.empty:
        features['has_subscription_history'] = 1
        features['total_subscriptions'] = len(subscription_history)
        features['subs_tier'] = subscription_history['subscription_tier'].iloc[0] 
        features['sub_price_level'] = subscription_history['price.level'].iloc[0] 
        features['mult_subs'] = subscription_history['multiple.subs'].iloc[0]
        features['sub_location'] = subscription_history['location'].iloc[0]
        features['sub_section'] = subscription_history['section'].iloc[0]
        features['sub_package'] = subscription_history['package'].iloc[0]
        features['sub_season'] = subscription_history['season'].iloc[0]
        features['sub_seat_number'] = subscription_history['no.seats'].iloc[0]
        

    # Ticket purchase history
    ticket_history = tickets_df[tickets_df['account.id'] == account_id].copy()
    if not ticket_history.empty:
        features['total_tickets_purchased'] = len(ticket_history)
        features['seasons_attended'] = ticket_history['season'].nunique()
        features['total_tickets'] = ticket_history['no.seats'].sum()
        ticket_history['price.level_num'] = ticket_history['price.level'].apply(map_price_level)
        features['average_price_level'] = ticket_history['price.level_num'].mean(skipna=True)
        features['average_seats_per_ticket'] = ticket_history['no.seats'].mean()

    # Calculate frequency (average number of tickets per purchase)
        if features['total_tickets_purchased'] > 0:
            features['frequency'] = features['total_tickets'] / features['total_tickets_purchased']

    
    # Number of Donations by Patron
    account_info = account_df[account_df['account.id'] == account_id]
    if not account_info.empty:
        features['donation_history'] = account_info['no.donations.lifetime'].iloc[0] 
        features['donation_amount'] = account_info['amount.donated.lifetime'].iloc[0] 
        features['donation_amount_2013'] = account_info['amount.donated.2013'].iloc[0]
        features['billing_zip'] = account_info['billing.zip.code'].iloc[0]
        features['billing_city'] = account_info['billing.city'].iloc[0]
        features['shipping_zip'] = account_info['shipping.zip.code'].iloc[0]
        features['initial_donation'] = account_info['first.donated'].iloc[0] 

    return features