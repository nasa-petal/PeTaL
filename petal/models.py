from petal import db

# Item database table
class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(2000), nullable=True)
    image_url = db.Column(db.String(40), nullable=False, default='default.png')

    def __repr__(self):
        return f"Item('{self.description}','{self.image_url}')"
