
class Weight():
    def __init__(self, id, weight, company_id):
        self.id = id
        self.company_id = company_id
        self.weight = weight

    def __str__(self):
        return '%s: %s' % (self.company_id, self.weight)

class Company():
    def __init__(self, id, comp_name):
        self.id = id
        self.comp_name = comp_name

    def __str__(self):
        return self.comp_name