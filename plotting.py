
from matplotlib import pyplot as plt

plt.figure()
plt.plot(range(params.epochs),training_loss)
plt.title("Training Loss v/s Epochs")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")

# save figure with model name and epsilon2
plt.savefig("{}_{}_{}_Training Loss.png".format(params.model_name,args.data, args.eps2))

# create figure for test loss vs epochs
plt.figure()
plt.plot(range(args.epochs),val_loss)
plt.title("Test Loss v/s Epochs")
plt.xlabel("Epochs")
plt.ylabel("Test Loss")

# save figure with model name and epsilon2
plt.savefig("{}_{}_{}_Test Loss.png".format(args.model_name, args.data, args.eps2))

plt.figure()
plt.plot(range(args.epochs),tacc)
plt.title("Test Accuracy v/s Epochs")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")

plt.savefig("{}_{}_{}_TestAccuracy.png".format(args.model_name, args.data, args.eps2))

plt.figure()
plt.plot(range(args.epochs),tracc)
plt.title("Train Accuracy v/s Epochs")
plt.xlabel("Epochs")
plt.ylabel("Train Accuracy")

# save figure with model name and epsilon2
plt.savefig("{}_{}_{}_TrainAccuracy.png".format(args.model_name, args.data, args.eps2))

plt.figure()

e = 1
for grad in grad_dict:
    grad = grad.cpu().numpy()
    grad = grad[np.arange(0,grad.shape[0],1000)]
    plt.scatter(range(grad.shape[0]),grad,label="epoch:{}".format(50*e))
    e = e + 1
plt.legend()
plt.ylabel('absolute gradient')
    
plt.title("Absolute value of gradients")
plt.savefig("{}_{}_{}_Grad.png".format(args.model_name,args.data, args.eps2))

grad_img = read_image("{}_{}_{}_Grad.png".format(args.model_name,args.data, args.eps2)) 
grad_img = grad_img[:3]
sw.add_image('grad',grad_img)